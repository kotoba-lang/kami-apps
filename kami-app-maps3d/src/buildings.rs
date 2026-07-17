//! Building-extrude pipeline for `kami-app-maps3d`.
//!
//! Phase 1 scope: render building footprints from `getChunk`/`tileGeoJson`
//! XRPC as axis-aligned-bounding-box (AABB) extrudes — `[minX, minZ] ..
//! [maxX, maxZ]` swept from `base_y` to `base_y + height_m`. Earcut of
//! arbitrary polygon rings is deferred (covered by `kami-map`'s
//! `add_extrude_layer` for the 2D path; a 3D earcut adapter can be added
//! to `kami-pipelines` later when needed by more than one game crate).
//!
//! The adapter doubles as the **collider source** for `KamiApp`'s 3-axis
//! AABB sweep (`with_collider_probe`). `aabb_solid(min, max)` returns
//! `true` iff the player capsule overlaps any building box. Combined
//! with the terrain floor probe, the player walks on procedural ground
//! and is blocked by building walls.
//!
//! JS feeds boxes via the wasm-bindgen `set_buildings_json` extern (see
//! `lib.rs`). Each call **replaces** the active set so the host can
//! re-issue on every `getChunk` window change.
//!
//! Footprint JSON shape:
//! ```json
//! [
//!   { "minX": -10.0, "maxX": 10.0,
//!     "minZ": -10.0, "maxZ": 10.0,
//!     "baseY": 0.0,  "height": 24.0,
//!     "color": [0.78, 0.74, 0.69] }
//! ]
//! ```

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use hecs::World;
use kami_app::{Camera, RenderPipeline};
use kami_pipelines::{fog_from_sun, sun_from_time};
use kami_render::scene_pipelines::{VoxelPipeline, VoxelUniform};
use kami_render::RenderContext;
use serde::Deserialize;
use std::cell::RefCell;
use std::rc::Rc;
use wgpu::util::DeviceExt;

/// AABB extrude representing one building.
#[derive(Debug, Clone, Copy)]
pub struct BuildingBox {
    pub min_x: f32,
    pub max_x: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub base_y: f32,
    pub height: f32,
    pub color: [f32; 3],
}

impl BuildingBox {
    pub fn min(&self) -> Vec3 {
        Vec3::new(self.min_x, self.base_y, self.min_z)
    }
    pub fn max(&self) -> Vec3 {
        Vec3::new(self.max_x, self.base_y + self.height, self.max_z)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct BuildingBoxJson {
    #[serde(rename = "minX")]
    pub min_x: f32,
    #[serde(rename = "maxX")]
    pub max_x: f32,
    #[serde(rename = "minZ")]
    pub min_z: f32,
    #[serde(rename = "maxZ")]
    pub max_z: f32,
    #[serde(rename = "baseY", default)]
    pub base_y: f32,
    pub height: f32,
    #[serde(default = "default_color")]
    pub color: [f32; 3],
}

fn default_color() -> [f32; 3] {
    // Stylised stone-warm tone; close to OSM building default.
    [0.78, 0.74, 0.69]
}

impl From<BuildingBoxJson> for BuildingBox {
    fn from(j: BuildingBoxJson) -> Self {
        Self {
            min_x: j.min_x.min(j.max_x),
            max_x: j.min_x.max(j.max_x),
            min_z: j.min_z.min(j.max_z),
            max_z: j.min_z.max(j.max_z),
            base_y: j.base_y,
            height: j.height.max(0.0),
            color: j.color,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BuildingVertex {
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
    /// Latest authoritative box list. JS injects via `lib.rs::set_buildings_json`.
    boxes: RefCell<Vec<BuildingBox>>,
    /// Currently uploaded GPU buffers.
    gpu: RefCell<Option<Gpu>>,
    /// Bumped by `set_boxes`; checked in `prepare()` so the GPU buffer
    /// rebuild only runs when a new set has arrived.
    epoch: RefCell<u64>,
    uploaded_epoch: RefCell<u64>,
    fog_density: f32,
}

#[derive(Clone)]
pub struct BuildingExtrudeAdapter {
    inner: Rc<Shared>,
}

impl BuildingExtrudeAdapter {
    pub fn new(ctx: &RenderContext) -> Self {
        Self {
            inner: Rc::new(Shared {
                pipeline: VoxelPipeline::new(&ctx.device, ctx.format),
                device: ctx.device.clone(),
                boxes: RefCell::new(Vec::new()),
                gpu: RefCell::new(None),
                epoch: RefCell::new(0),
                uploaded_epoch: RefCell::new(0),
                fog_density: 0.0012,
            }),
        }
    }

    /// Replace the active building set. Triggers a GPU rebuild on the
    /// next `prepare()` tick.
    pub fn set_boxes(&self, boxes: Vec<BuildingBox>) {
        *self.inner.boxes.borrow_mut() = boxes;
        *self.inner.epoch.borrow_mut() += 1;
    }

    /// `true` if the AABB `[min, max]` overlaps any active building.
    /// Used by `KamiApp::with_collider_probe`.
    pub fn aabb_solid(&self, min: Vec3, max: Vec3) -> bool {
        let boxes = self.inner.boxes.borrow();
        for b in boxes.iter() {
            let bmin = b.min();
            let bmax = b.max();
            // AABB-vs-AABB overlap: separated on any axis ⇒ no hit.
            if max.x <= bmin.x || min.x >= bmax.x { continue; }
            if max.y <= bmin.y || min.y >= bmax.y { continue; }
            if max.z <= bmin.z || min.z >= bmax.z { continue; }
            return true;
        }
        false
    }

    /// Highest building roof (y) at world XZ, or `None` if XZ is not
    /// over any building footprint. Combined with terrain probe in
    /// `lib.rs` so the player can walk on rooftops.
    pub fn rooftop_y(&self, world: Vec3) -> Option<f32> {
        let boxes = self.inner.boxes.borrow();
        let mut best: Option<f32> = None;
        for b in boxes.iter() {
            if world.x < b.min_x || world.x > b.max_x { continue; }
            if world.z < b.min_z || world.z > b.max_z { continue; }
            let top = b.base_y + b.height;
            if best.map_or(true, |y| top > y) {
                best = Some(top);
            }
        }
        best
    }

    fn rebuild_gpu(&self) {
        let boxes = self.inner.boxes.borrow();
        if boxes.is_empty() {
            *self.inner.gpu.borrow_mut() = None;
            return;
        }
        // Each box = 24 verts (4 per face × 6 faces, faceted normals)
        // and 36 indices.
        let mut verts: Vec<BuildingVertex> = Vec::with_capacity(boxes.len() * 24);
        let mut indices: Vec<u32> = Vec::with_capacity(boxes.len() * 36);
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            // +X
            ([1.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
            // -X
            ([-1.0, 0.0, 0.0], [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            // +Y (roof)
            ([0.0, 1.0, 0.0], [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
            // -Y (ground — typically hidden by terrain)
            ([0.0, -1.0, 0.0], [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
            // +Z
            ([0.0, 0.0, 1.0], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            // -Z
            ([0.0, 0.0, -1.0], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        ];
        for b in boxes.iter() {
            let dx = b.max_x - b.min_x;
            let dz = b.max_z - b.min_z;
            let dy = b.height;
            // Subtle roof tint so flat-shaded boxes read as buildings,
            // not voxel chunks.
            let roof_tint = [b.color[0] * 0.92, b.color[1] * 0.92, b.color[2] * 0.95];
            for (face_i, (n, corners)) in faces.iter().enumerate() {
                let base_idx = verts.len() as u32;
                let col = if face_i == 2 { roof_tint } else { b.color };
                for c in corners {
                    let p = [
                        b.min_x + c[0] * dx,
                        b.base_y + c[1] * dy,
                        b.min_z + c[2] * dz,
                    ];
                    verts.push(BuildingVertex { pos: p, norm: *n, col });
                }
                indices.extend_from_slice(&[
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx, base_idx + 2, base_idx + 3,
                ]);
            }
        }
        let vb = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.buildings.vb"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.buildings.ib"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        *self.inner.gpu.borrow_mut() = Some(Gpu {
            vb,
            ib,
            index_count: indices.len() as u32,
        });
    }
}

impl RenderPipeline for BuildingExtrudeAdapter {
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
        let Some(gpu) = gpu.as_ref() else { return };

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
            label: Some("maps3d.buildings"),
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
