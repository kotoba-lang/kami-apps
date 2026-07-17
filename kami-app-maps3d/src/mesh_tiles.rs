//! Photogrammetry mesh tiles — render-only adapter.
//!
//! Consumes vertex-colored GLB tiles produced by the
//! `maps3d.simplifyAndExport` BPMN task (Open3D quadric_decimation +
//! saturated vertex colors). Each tile = one H3 res-12 cell ≈ 9 m edge,
//! ~5 K triangles after simplification. Render uses the same
//! `VoxelPipeline` (pos3 + norm3 + col3 vertex layout) as the building
//! AABB extrudes, so no engine pipeline addition is required.
//!
//! **Render-only by design** — collision stays on the OSM AABB
//! `BuildingExtrudeAdapter`. Photogrammetry geometry is the *visual
//! skin*; OSM footprints are the authoritative *collision shape*. This
//! avoids per-triangle SAT or rasterizing meshes into occupancy grids.
//!
//! Per-tile upserts (`upsert_tile`) replace any prior GPU buffers for
//! that H3 cell. Removing a tile (`remove_tile`) drops the GPU memory.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use hecs::World;
use kami_app::{Camera, RenderPipeline};
use kami_pipelines::{fog_from_sun, sun_from_time};
use kami_render::scene_pipelines::{VoxelPipeline, VoxelUniform};
use kami_render::RenderContext;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TileVertex {
    pos: [f32; 3],
    norm: [f32; 3],
    col: [f32; 3],
}

struct GpuMesh {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    index_count: u32,
}

struct Shared {
    pipeline: VoxelPipeline,
    device: wgpu::Device,
    tiles: RefCell<HashMap<String, GpuMesh>>,
    fog_density: f32,
}

#[derive(Clone)]
pub struct MeshTileAdapter {
    inner: Rc<Shared>,
}

#[derive(Debug)]
pub enum MeshTileError {
    Gltf(String),
    NoMesh,
    Format(String),
}

impl std::fmt::Display for MeshTileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshTileError::Gltf(m) => write!(f, "gltf: {m}"),
            MeshTileError::NoMesh => write!(f, "gltf: no mesh in scene"),
            MeshTileError::Format(m) => write!(f, "format: {m}"),
        }
    }
}

impl MeshTileAdapter {
    pub fn new(ctx: &RenderContext) -> Self {
        Self {
            inner: Rc::new(Shared {
                pipeline: VoxelPipeline::new(&ctx.device, ctx.format),
                device: ctx.device.clone(),
                tiles: RefCell::new(HashMap::new()),
                fog_density: 0.0012,
            }),
        }
    }

    /// Replace (or insert) the GPU buffers for `tile_h3` from a GLB byte
    /// slice. Picks **primitive 0 of mesh 0** — multi-primitive tiles
    /// would need a follow-up. Vertex layout is normalised to
    /// pos3+norm3+col3 (col defaults to mid-grey if the GLB lacks
    /// COLOR_0; norm defaults to (0,1,0) if NORMAL is absent).
    pub fn upsert_tile(&self, tile_h3: &str, glb_bytes: &[u8]) -> Result<(), MeshTileError> {
        let data = parse_mesh_data(glb_bytes)?;
        let verts: Vec<TileVertex> = data
            .positions
            .iter()
            .zip(data.normals.iter())
            .zip(data.colors.iter())
            .map(|((p, n), c)| TileVertex {
                pos: *p,
                norm: *n,
                col: *c,
            })
            .collect();

        let vb = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.tile.vb"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maps3d.tile.ib"),
                contents: bytemuck::cast_slice(&data.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.inner.tiles.borrow_mut().insert(
            tile_h3.to_string(),
            GpuMesh {
                vb,
                ib,
                index_count: data.indices.len() as u32,
            },
        );
        Ok(())
    }

    pub fn remove_tile(&self, tile_h3: &str) {
        self.inner.tiles.borrow_mut().remove(tile_h3);
    }

    pub fn tile_count(&self) -> usize {
        self.inner.tiles.borrow().len()
    }
}

fn pad_or_truncate<T: Clone>(mut v: Vec<T>, n: usize, fill: T) -> Vec<T> {
    if v.len() == n {
        return v;
    }
    if v.len() > n {
        v.truncate(n);
        return v;
    }
    while v.len() < n {
        v.push(fill.clone());
    }
    v
}

fn parse_glb(
    bytes: &[u8],
) -> Result<(gltf::Document, Vec<gltf::buffer::Data>), MeshTileError> {
    let (doc, buffers, _images) =
        gltf::import_slice(bytes).map_err(|e| MeshTileError::Gltf(e.to_string()))?;
    Ok((doc, buffers))
}

/// GPU-free representation of a parsed GLB tile. Used internally by
/// `upsert_tile` and exposed publicly so tests + other callers can run
/// the parser without a wgpu device.
#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

/// Parse the first primitive of mesh 0 in a GLB byte slice into
/// pos+normal+color+indices. Defaults: normal = (0,1,0), color =
/// stone-warm grey when the source GLB omits them.
pub fn parse_mesh_data(glb_bytes: &[u8]) -> Result<MeshData, MeshTileError> {
    let (doc, buffers) = parse_glb(glb_bytes)?;
    let mesh = doc.meshes().next().ok_or(MeshTileError::NoMesh)?;
    let prim = mesh.primitives().next().ok_or(MeshTileError::NoMesh)?;
    let reader = prim.reader(|b| Some(&buffers[b.index()][..]));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| MeshTileError::Format("missing POSITION".into()))?
        .collect();
    let n = positions.len();
    if n == 0 {
        return Err(MeshTileError::Format("empty primitive".into()));
    }

    let normals: Vec<[f32; 3]> = match reader.read_normals() {
        Some(it) => it.collect(),
        None => vec![[0.0, 1.0, 0.0]; n],
    };
    let colors: Vec<[f32; 3]> = match reader.read_colors(0) {
        Some(c) => c.into_rgb_f32().collect(),
        None => vec![[0.78, 0.74, 0.69]; n],
    };
    let normals = pad_or_truncate(normals, n, [0.0, 1.0, 0.0]);
    let colors = pad_or_truncate(colors, n, [0.78, 0.74, 0.69]);

    let indices: Vec<u32> = match reader.read_indices() {
        Some(it) => it.into_u32().collect(),
        None => (0..n as u32).collect(),
    };
    if indices.is_empty() {
        return Err(MeshTileError::Format("empty index buffer".into()));
    }
    Ok(MeshData {
        positions,
        normals,
        colors,
        indices,
    })
}

impl RenderPipeline for MeshTileAdapter {
    fn prepare(&mut self, _ctx: &RenderContext, _camera: &Camera, _world: &World) {}

    fn record(
        &self,
        ctx: &RenderContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: &Camera,
        _world: &World,
    ) {
        let tiles = self.inner.tiles.borrow();
        if tiles.is_empty() {
            return;
        }

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
            label: Some("maps3d.tiles"),
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
        for tile in tiles.values() {
            pass.set_vertex_buffer(0, tile.vb.slice(..));
            pass.set_index_buffer(tile.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..tile.index_count, 0, 0..1);
        }
    }
}

// Vec3 silence for non-wasm builds where the import path differs.
#[allow(dead_code)]
fn _vec3_silence(_v: Vec3) {}

// ───────────────────────────────────────────────────────────────────
// Tests — GPU-free, run via `cargo test -p kami-app-maps3d`.
// ───────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a minimal GLB 2.0 binary in-memory: 3 positions
    /// (single triangle), 3 normals, 3 RGB colors, 3 indices.
    /// `with_normals` / `with_colors` toggle whether those attributes
    /// + their bufferViews are emitted, so the test can exercise the
    /// default-fallback branches.
    fn synth_glb_triangle() -> Vec<u8> {
        synth_glb_triangle_attr(true, true)
    }

    fn synth_glb_triangle_attr(with_normals: bool, with_colors: bool) -> Vec<u8> {
        let positions: [[f32; 3]; 3] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals: [[f32; 3]; 3] = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let colors: [[f32; 3]; 3] = [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
            [0.70, 0.80, 0.90],
        ];
        let indices: [u16; 3] = [0, 1, 2];

        let pos_bytes = 36usize;
        let norm_bytes = 36usize;
        let col_bytes = 36usize;
        let mut bin: Vec<u8> = Vec::new();
        for p in &positions { for v in p { bin.extend_from_slice(&v.to_le_bytes()); } }
        if with_normals { for p in &normals { for v in p { bin.extend_from_slice(&v.to_le_bytes()); } } }
        if with_colors  { for p in &colors  { for v in p { bin.extend_from_slice(&v.to_le_bytes()); } } }
        let idx_offset = bin.len();
        for i in &indices { bin.extend_from_slice(&i.to_le_bytes()); }
        while bin.len() % 4 != 0 { bin.push(0); }
        let bin_byte_length = bin.len();

        // Build attributes + accessors + bufferViews dynamically so we
        // can omit NORMAL / COLOR_0 cleanly.
        let mut attributes = serde_json::Map::new();
        attributes.insert("POSITION".into(), serde_json::Value::from(0u32));
        let mut accessors = vec![
            serde_json::json!({ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                                "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] }),
        ];
        let mut buffer_views = vec![
            serde_json::json!({ "buffer": 0, "byteOffset": 0, "byteLength": pos_bytes, "target": 34962 }),
        ];
        let mut byte_cursor = pos_bytes;
        if with_normals {
            let bv_idx = buffer_views.len();
            buffer_views.push(serde_json::json!({
                "buffer": 0, "byteOffset": byte_cursor, "byteLength": norm_bytes, "target": 34962
            }));
            accessors.push(serde_json::json!({
                "bufferView": bv_idx, "componentType": 5126, "count": 3, "type": "VEC3"
            }));
            attributes.insert("NORMAL".into(), serde_json::Value::from(accessors.len() as u32 - 1));
            byte_cursor += norm_bytes;
        }
        if with_colors {
            let bv_idx = buffer_views.len();
            buffer_views.push(serde_json::json!({
                "buffer": 0, "byteOffset": byte_cursor, "byteLength": col_bytes, "target": 34962
            }));
            accessors.push(serde_json::json!({
                "bufferView": bv_idx, "componentType": 5126, "count": 3, "type": "VEC3"
            }));
            attributes.insert("COLOR_0".into(), serde_json::Value::from(accessors.len() as u32 - 1));
            byte_cursor += col_bytes;
        }
        // Indices buffer view + accessor (always last).
        let idx_bv = buffer_views.len();
        buffer_views.push(serde_json::json!({
            "buffer": 0, "byteOffset": idx_offset, "byteLength": 6, "target": 34963
        }));
        accessors.push(serde_json::json!({
            "bufferView": idx_bv, "componentType": 5123, "count": 3, "type": "SCALAR"
        }));
        let idx_accessor = accessors.len() as u32 - 1;
        let _ = byte_cursor;

        let json = serde_json::json!({
            "asset": { "version": "2.0", "generator": "kami-app-maps3d-test" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": serde_json::Value::Object(attributes),
                    "indices": idx_accessor,
                    "mode": 4
                }]
            }],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{ "byteLength": bin_byte_length }]
        });
        let mut json_bytes = serde_json::to_vec(&json).expect("serialize gltf json");
        // GLB JSON chunk must be 4-byte aligned, padded with 0x20.
        while json_bytes.len() % 4 != 0 { json_bytes.push(0x20); }

        // GLB header (12 B) + JSON chunk (8 B header + json_bytes) +
        // BIN chunk (8 B header + bin).
        let total_len = 12 + 8 + json_bytes.len() + 8 + bin_byte_length;
        let mut out: Vec<u8> = Vec::with_capacity(total_len);
        out.extend_from_slice(b"glTF");                     // magic
        out.extend_from_slice(&2u32.to_le_bytes());         // version
        out.extend_from_slice(&(total_len as u32).to_le_bytes());
        // JSON chunk.
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json_bytes);
        // BIN chunk.
        out.extend_from_slice(&(bin_byte_length as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
        assert_eq!(out.len(), total_len);
        out
    }

    #[test]
    fn parses_synthetic_triangle_glb() {
        let glb = synth_glb_triangle();
        let data = parse_mesh_data(&glb).expect("parse GLB");
        assert_eq!(data.positions.len(), 3, "vertex count");
        assert_eq!(data.normals.len(), 3, "normal count");
        assert_eq!(data.colors.len(), 3, "color count");
        assert_eq!(data.indices, vec![0u32, 1, 2]);
        assert_eq!(data.positions[1], [1.0, 0.0, 0.0]);
        assert_eq!(data.normals[2], [0.0, 0.0, 1.0]);
        // Color 1 was [0.40, 0.50, 0.60]; allow tiny f32 round-trip.
        let c = data.colors[1];
        assert!((c[0] - 0.40).abs() < 1e-5);
        assert!((c[1] - 0.50).abs() < 1e-5);
        assert!((c[2] - 0.60).abs() < 1e-5);
    }

    #[test]
    fn rejects_garbage_glb() {
        let err = parse_mesh_data(b"not a glb").err().expect("must error");
        match err {
            MeshTileError::Gltf(_) => {}
            other => panic!("expected Gltf error, got {other:?}"),
        }
    }

    #[test]
    fn defaults_normals_and_colors_when_missing() {
        // Synthesize a GLB that omits NORMAL + COLOR_0 entirely. The
        // parser must fall back to (0,1,0) and the stone-warm default
        // grey.
        let glb = synth_glb_triangle_attr(/*normals=*/ false, /*colors=*/ false);
        let data = parse_mesh_data(&glb).expect("parse w/ defaults");
        assert_eq!(data.positions.len(), 3);
        assert_eq!(data.normals.len(), 3);
        assert_eq!(data.colors.len(), 3);
        for n in &data.normals {
            assert_eq!(n, &[0.0, 1.0, 0.0]);
        }
        for c in &data.colors {
            assert!((c[0] - 0.78).abs() < 1e-5);
            assert!((c[1] - 0.74).abs() < 1e-5);
            assert!((c[2] - 0.69).abs() < 1e-5);
        }
    }
}
