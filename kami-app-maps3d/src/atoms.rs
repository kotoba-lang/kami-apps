//! CPK atom sphere rendering for maps3d.
//!
//! JS calls `set_atoms_json` with an array of atomic positions from
//! `com.etzhayyim.apps.maps.getChunk`. Each entry carries element symbol,
//! CPK colour, physical sphere radius (pm), and world-space position.
//!
//! Spheres are UV-sphere meshes (8 stacks × 16 slices). Radius is
//! converted `sphere_r_pm × 0.001` → world units so a 150 pm atom
//! appears at 0.15 m diameter, visible at human scale.
//!
//! ```js
//! set_atoms_json(JSON.stringify([
//!   { symbol: "C",  colorR: 0.20, colorG: 0.20, colorB: 0.20,
//!     sphereRPm: 77.0,  worldX: 0.0, worldY: 1.5, worldZ: 0.0 },
//!   { symbol: "O",  colorR: 0.85, colorG: 0.10, colorB: 0.10,
//!     sphereRPm: 73.0,  worldX: 1.2, worldY: 1.5, worldZ: 0.0 }
//! ]));
//! ```

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use hecs::World;
use kami_app::{Camera, RenderPipeline};
use kami_pipelines::{fog_from_sun, sun_from_time};
use kami_render::scene_pipelines::{VoxelPipeline, VoxelUniform};
use kami_render::RenderContext;
use serde::Deserialize;
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use wgpu::util::DeviceExt;

/// sphere_r_pm → world-unit radius conversion.
const PM_TO_WORLD: f32 = 0.001;

/// UV-sphere resolution.
const STACKS: u32 = 8;
const SLICES: u32 = 16;

#[derive(Debug, Deserialize)]
pub struct AtomItemJson {
    pub symbol: String,
    #[serde(rename = "colorR")]
    pub color_r: f32,
    #[serde(rename = "colorG")]
    pub color_g: f32,
    #[serde(rename = "colorB")]
    pub color_b: f32,
    /// Physical covalent / van-der-Waals radius in picometers.
    /// Converted to world units via `× 0.001`.
    #[serde(rename = "sphereRPm")]
    pub sphere_r_pm: f32,
    #[serde(rename = "worldX")]
    pub world_x: f32,
    #[serde(rename = "worldY")]
    pub world_y: f32,
    #[serde(rename = "worldZ")]
    pub world_z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AtomVertex {
    pos: [f32; 3],
    norm: [f32; 3],
    col: [f32; 3],
}

/// Generate a UV-sphere centred at (cx, cy, cz) with radius r.
fn gen_sphere(
    cx: f32,
    cy: f32,
    cz: f32,
    r: f32,
    col: [f32; 3],
    base: u32,
) -> (Vec<AtomVertex>, Vec<u32>) {
    let rows = STACKS + 1;
    let cols = SLICES + 1;
    let mut verts: Vec<AtomVertex> = Vec::with_capacity((rows * cols) as usize);
    let mut idxs: Vec<u32> = Vec::with_capacity((STACKS * SLICES * 6) as usize);

    for s in 0..rows {
        let phi = PI * (s as f32) / (STACKS as f32);
        let (sin_phi, cos_phi) = phi.sin_cos();
        for l in 0..cols {
            let theta = 2.0 * PI * (l as f32) / (SLICES as f32);
            let (sin_theta, cos_theta) = theta.sin_cos();
            let nx = sin_phi * cos_theta;
            let ny = cos_phi;
            let nz = sin_phi * sin_theta;
            verts.push(AtomVertex {
                pos: [cx + r * nx, cy + r * ny, cz + r * nz],
                norm: [nx, ny, nz],
                col,
            });
        }
    }

    for s in 0..STACKS {
        for l in 0..SLICES {
            let cur = base + s * cols + l;
            let nxt = cur + cols;
            idxs.extend_from_slice(&[cur, nxt, cur + 1, cur + 1, nxt, nxt + 1]);
        }
    }

    (verts, idxs)
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
    items: RefCell<Vec<AtomItemJson>>,
    gpu: RefCell<Option<Gpu>>,
    fog_density: f32,
}

#[derive(Clone)]
pub struct AtomAdapter {
    inner: Rc<Shared>,
}

impl AtomAdapter {
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

    pub fn set_items(&self, items: Vec<AtomItemJson>) {
        *self.inner.items.borrow_mut() = items;
        *self.inner.epoch.borrow_mut() += 1;
    }

    fn rebuild_gpu(&self) {
        let items = self.inner.items.borrow();
        if items.is_empty() {
            *self.inner.gpu.borrow_mut() = None;
            return;
        }

        let mut all_verts: Vec<AtomVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for atom in items.iter() {
            let r = (atom.sphere_r_pm * PM_TO_WORLD).max(0.02);
            let col = [atom.color_r, atom.color_g, atom.color_b];
            let base = all_verts.len() as u32;
            let (v, i) = gen_sphere(atom.world_x, atom.world_y, atom.world_z, r, col, base);
            all_verts.extend(v);
            all_indices.extend(i);
        }

        if all_verts.is_empty() {
            *self.inner.gpu.borrow_mut() = None;
            return;
        }

        let vb = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.atoms.vb"),
                contents: bytemuck::cast_slice(&all_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.atoms.ib"),
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

impl RenderPipeline for AtomAdapter {
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
            label: Some("maps3d.atoms"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_vertex_count() {
        let (v, i) = gen_sphere(0.0, 0.0, 0.0, 1.0, [1.0; 3], 0);
        assert_eq!(v.len() as u32, (STACKS + 1) * (SLICES + 1));
        assert_eq!(i.len() as u32, STACKS * SLICES * 6);
    }

    #[test]
    fn sphere_normals_unit_length() {
        let (v, _) = gen_sphere(0.0, 2.0, 0.0, 0.5, [0.0; 3], 0);
        for vert in &v {
            let n = vert.norm;
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "norm not unit: {len}");
        }
    }

    #[test]
    fn atom_json_round_trip() {
        let json = r#"[{
            "symbol":"O","colorR":0.85,"colorG":0.10,"colorB":0.10,
            "sphereRPm":73.0,"worldX":1.2,"worldY":0.0,"worldZ":0.0
        }]"#;
        let items: Vec<AtomItemJson> = serde_json::from_str(json).expect("parse");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].symbol, "O");
        assert!((items[0].sphere_r_pm - 73.0).abs() < 1e-5);
    }
}
