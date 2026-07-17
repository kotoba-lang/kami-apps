//! DB-driven vegetation placement for maps3d.
//!
//! JS calls `set_vegetation_json` with positioned vegetation items from
//! `com.etzhayyim.apps.maps.getChunk`. Each item carries a JSON-serialised
//! `OwnedTaxonomicProfile` (from `vertex_scientific_taxon`) plus a
//! world-space position and scale.
//!
//! `kami-vegetation` emits pos3+uv2 (5 f32/vert); this adapter converts
//! to pos3+norm3+col3 so it can share the existing `VoxelPipeline`.
//!
//! ```js
//! set_vegetation_json(JSON.stringify([
//!   {
//!     renderProfileJson: '{"canopy":"Blade","division":"angiospermae",...}',
//!     worldX: 12.0, worldY: 0.0, worldZ: -8.5,
//!     scaleX: 1.0, scaleY: 1.5, scaleZ: 1.0
//!   }
//! ]));
//! ```

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use hecs::World;
use kami_app::{Camera, RenderPipeline};
use kami_pipelines::{fog_from_sun, sun_from_time};
use kami_render::scene_pipelines::{VoxelPipeline, VoxelUniform};
use kami_render::RenderContext;
use kami_vegetation::mesh::mesh_from_profile;
use kami_vegetation::taxonomy::{
    CanopyShape, Division, GrowthHabit, LeafArrangement, LeafShape, OwnedTaxonomicProfile,
    TaxonomicProfile,
};
use serde::Deserialize;
use std::cell::RefCell;
use std::rc::Rc;
use wgpu::util::DeviceExt;

#[derive(Debug, Deserialize)]
pub struct VegetationItemJson {
    #[serde(rename = "renderProfileJson")]
    pub render_profile_json: String,
    #[serde(rename = "worldX")]
    pub world_x: f32,
    #[serde(rename = "worldY")]
    pub world_y: f32,
    #[serde(rename = "worldZ")]
    pub world_z: f32,
    #[serde(rename = "scaleX", default = "one")]
    pub scale_x: f32,
    #[serde(rename = "scaleY", default = "one")]
    pub scale_y: f32,
    #[serde(rename = "scaleZ", default = "one")]
    pub scale_z: f32,
}

fn one() -> f32 {
    1.0
}

/// Convert `OwnedTaxonomicProfile` → `TaxonomicProfile`.
/// Box::leak for common_name is acceptable — distinct species names are
/// bounded by the taxonomy DB size (typically < 1 000).
fn owned_to_profile(o: &OwnedTaxonomicProfile) -> TaxonomicProfile {
    let leaked: &'static str = Box::leak(o.common_name.clone().into_boxed_str());
    TaxonomicProfile {
        common_name: leaked,
        division: o.division,
        habit: o.habit,
        arrangement: o.arrangement,
        leaf_shape: o.leaf_shape,
        canopy: o.canopy,
        height_range: o.height_range,
        stem_radius_base: o.stem_radius_base,
        stem_radius_top: o.stem_radius_top,
        leaf_count: o.leaf_count,
        leaf_size: o.leaf_size,
        color_base: o.color_base,
        color_tip: o.color_tip,
    }
}

/// Canopy-type → base display colour (RGB linear).
fn canopy_color(shape: CanopyShape) -> [f32; 3] {
    match shape {
        CanopyShape::Blade  => [0.25, 0.55, 0.10], // grass
        CanopyShape::Fan    => [0.18, 0.50, 0.18], // fern
        CanopyShape::Radial => [0.18, 0.52, 0.22], // palm
        CanopyShape::Cone   => [0.12, 0.38, 0.22], // conifer
        CanopyShape::Dome   => [0.28, 0.52, 0.18], // bush
        CanopyShape::Column => [0.30, 0.38, 0.16], // cactus
        CanopyShape::Carpet => [0.35, 0.52, 0.22], // moss
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VegVertex {
    pos: [f32; 3],
    norm: [f32; 3],
    col: [f32; 3],
}

struct Gpu {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    index_count: u32,
}

struct Shared {
    pipeline: VoxelPipeline,
    device: wgpu::Device,
    epoch: RefCell<u64>,
    uploaded_epoch: RefCell<u64>,
    items: RefCell<Vec<VegetationItemJson>>,
    gpu: RefCell<Option<Gpu>>,
    fog_density: f32,
}

#[derive(Clone)]
pub struct VegetationAdapter {
    inner: Rc<Shared>,
}

impl VegetationAdapter {
    pub fn new(ctx: &RenderContext) -> Self {
        Self {
            inner: Rc::new(Shared {
                pipeline: VoxelPipeline::new(&ctx.device, ctx.format),
                device: ctx.device.clone(),
                epoch: RefCell::new(0),
                uploaded_epoch: RefCell::new(0),
                items: RefCell::new(Vec::new()),
                gpu: RefCell::new(None),
                fog_density: 0.0012,
            }),
        }
    }

    pub fn set_items(&self, items: Vec<VegetationItemJson>) {
        *self.inner.items.borrow_mut() = items;
        *self.inner.epoch.borrow_mut() += 1;
    }

    fn rebuild_gpu(&self) {
        let items = self.inner.items.borrow();
        if items.is_empty() {
            *self.inner.gpu.borrow_mut() = None;
            return;
        }

        let mut all_verts: Vec<VegVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for item in items.iter() {
            let owned = match OwnedTaxonomicProfile::from_json_str(&item.render_profile_json) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let profile = owned_to_profile(&owned);
            let spec_mesh = mesh_from_profile(&profile);
            let col = canopy_color(profile.canopy);

            let trs = Mat4::from_scale_rotation_translation(
                Vec3::new(item.scale_x, item.scale_y, item.scale_z),
                glam::Quat::IDENTITY,
                Vec3::new(item.world_x, item.world_y, item.world_z),
            );

            // Extract world positions (stride 5 — pos3 + uv2).
            let n_verts = spec_mesh.vertex_count as usize;
            let mut world_pos: Vec<Vec3> = Vec::with_capacity(n_verts);
            for vi in 0..n_verts {
                let off = vi * 5;
                let lp = Vec3::new(
                    spec_mesh.vertices[off],
                    spec_mesh.vertices[off + 1],
                    spec_mesh.vertices[off + 2],
                );
                world_pos.push(trs.transform_point3(lp));
            }

            // Accumulate face normals per vertex.
            let mut normals = vec![Vec3::ZERO; n_verts];
            for tri in spec_mesh.indices.chunks_exact(3) {
                let (ai, bi, ci) =
                    (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let fn_ = (world_pos[bi] - world_pos[ai])
                    .cross(world_pos[ci] - world_pos[ai]);
                normals[ai] += fn_;
                normals[bi] += fn_;
                normals[ci] += fn_;
            }

            let base = all_verts.len() as u32;
            for vi in 0..n_verts {
                all_verts.push(VegVertex {
                    pos: world_pos[vi].to_array(),
                    norm: normals[vi].normalize_or_zero().to_array(),
                    col,
                });
            }
            for &idx in &spec_mesh.indices {
                all_indices.push(base + idx);
            }
        }

        if all_verts.is_empty() {
            *self.inner.gpu.borrow_mut() = None;
            return;
        }

        let vb = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.veg.vb"),
                contents: bytemuck::cast_slice(&all_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.veg.ib"),
                contents: bytemuck::cast_slice(&all_indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        *self.inner.gpu.borrow_mut() = Some(Gpu {
            vb,
            ib,
            index_count: all_indices.len() as u32,
        });
    }
}

impl RenderPipeline for VegetationAdapter {
    fn prepare(&mut self, _ctx: &RenderContext, _camera: &Camera, _world: &World) {
        let epoch = *self.inner.epoch.borrow();
        let uploaded = *self.inner.uploaded_epoch.borrow();
        if epoch != uploaded {
            self.rebuild_gpu();
            *self.inner.uploaded_epoch.borrow_mut() = epoch;
        }
    }

    fn record(
        &self,
        ctx: &RenderContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &Camera,
        _world: &World,
    ) {
        let gpu = self.inner.gpu.borrow();
        let Some(gpu) = gpu.as_ref() else {
            return;
        };

        let u = camera.as_render().uniform();
        let view_m = Mat4::from_cols_array_2d(&u.view);
        let proj = Mat4::from_cols_array_2d(&u.projection);
        let vp = proj * view_m;
        let sun_dir = sun_from_time(camera.time);
        let fog = fog_from_sun(sun_dir);
        let warmth = 1.0 - sun_dir.y.max(0.0);
        let sun_color = [1.0, 0.96 - warmth * 0.12, 0.88 - warmth * 0.28];
        let vu = VoxelUniform {
            view_proj: vp.to_cols_array(),
            cam_pos: u.position,
            _p0: 0.0,
            sun_dir: sun_dir.to_array(),
            _p1: 0.0,
            sun_color,
            fog_density: self.inner.fog_density,
            fog_color: fog.to_array(),
            _p2: 0.0,
        };
        ctx.queue
            .write_buffer(&self.inner.pipeline.uniform, 0, bytemuck::bytes_of(&vu));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("maps3d.veg"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.inner.pipeline.pipeline);
        pass.set_bind_group(0, &self.inner.pipeline.bind_group, &[]);
        pass.set_vertex_buffer(0, gpu.vb.slice(..));
        pass.set_index_buffer(gpu.ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..gpu.index_count, 0, 0..1);
    }
}

// Suppress dead_code warnings for non-wasm targets (tests / cargo check).
#[allow(dead_code)]
fn _type_silence(
    _d: Division,
    _h: GrowthHabit,
    _a: LeafArrangement,
    _l: LeafShape,
) {
}
