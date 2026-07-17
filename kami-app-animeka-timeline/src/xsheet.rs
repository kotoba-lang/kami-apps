//! X-sheet (タイムシート) grid pipeline.
//!
//! Renders a static frame × lane grid using instanced quads:
//!
//! ```text
//! frame | A    | B    | C    | dialogue    | camera
//! -----+------+------+------+-------------+--------
//!    1 |  ●   |  ○   |      |             |
//!    2 |      |      |      | 明日、どうな  | TU 2s
//!    3 |  ●   |  ○   |  ●   |             |
//!   ...
//! ```
//!
//! Phase 1 (this file): draws the grid background + alternating row bands +
//! 1-second (every `fps` frames) accent rows. Per-cell stroke/text/dialogue
//! content is left for Phase 2 — those require a text atlas (`kami-text`) +
//! instance data stream keyed by `(frame, lane)`.

use bytemuck::{Pod, Zeroable};
use hecs::World;
use kami_app::{Camera, GpuCtx, RenderPipeline};

use crate::QuadVertex;

/// Uniform block — just the ortho projection; extend as needed (time, playhead).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct XSheetUniforms {
    view_proj: [[f32; 4]; 4],
    playhead_frame: f32,
    _pad: [f32; 3],
}

const WGSL: &str = r#"
struct Uniforms { view_proj: mat4x4<f32>, playhead_frame: f32, _pad0: f32, _pad1: f32, _pad2: f32 };
@group(0) @binding(0) var<uniform> U: Uniforms;

struct VIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv:  vec2<f32>,
    @location(2) rgba: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) rgba: vec4<f32>,
};

@vertex
fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    out.clip = U.view_proj * vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.rgba = in.rgba;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return in.rgba;
}
"#;

pub struct XSheetPipeline {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    // Phase 2: per-frame playhead + per-cell text overlay need these.
    #[allow(dead_code)]
    frame_count: u32,
    #[allow(dead_code)]
    lanes: u32,
    #[allow(dead_code)]
    fps: u32,
}

impl XSheetPipeline {
    /// World-space row height (one frame = one row). Keep in sync with
    /// `run_animeka_timeline`'s camera extent derivation.
    pub const ROW_HEIGHT: f32 = 1.0;

    /// Column widths (world units) for: frame# | A | B | C | dialogue | camera.
    pub const COL_WIDTHS: [f32; 6] = [1.2, 1.6, 1.6, 1.6, 4.0, 2.4];

    pub fn new(ctx: &GpuCtx, frame_count: u32, lanes: u32, fps: u32) -> Self {
        let device = &ctx.device;

        // Build static grid geometry.
        let (verts, idxs) = build_grid_mesh(frame_count, lanes, fps);

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xsheet-vtx"),
            size: (verts.len() * std::mem::size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&verts));

        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xsheet-idx"),
            size: (idxs.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&index_buf, 0, bytemuck::cast_slice(&idxs));

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xsheet-uni"),
            size: std::mem::size_of::<XSheetUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xsheet-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xsheet-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xsheet-wgsl"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xsheet-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("xsheet-rp"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[QuadVertex::LAYOUT],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            vertex_buf,
            index_buf,
            index_count: idxs.len() as u32,
            uniform_buf,
            bind_group,
            pipeline,
            frame_count,
            lanes,
            fps,
        }
    }
}

impl RenderPipeline for XSheetPipeline {
    fn prepare(&mut self, ctx: &GpuCtx, camera: &Camera, _world: &World) {
        let uni = XSheetUniforms {
            view_proj: camera.view_projection().to_cols_array_2d(),
            playhead_frame: 0.0,
            _pad: [0.0; 3],
        };
        ctx.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uni));
    }

    fn record(
        &self,
        _ctx: &GpuCtx,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _depth_view: &wgpu::TextureView,
        _camera: &Camera,
        _world: &World,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("xsheet-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Nintendo cream background (40-engine/kami-engine/CLAUDE.md UI/UX).
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.941, g: 0.917, b: 0.839, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Build the grid mesh: row bands (alternating shade) + 1-second accent rows
/// + column dividers. All geometry is centered on origin in world space.
fn build_grid_mesh(frame_count: u32, _lanes: u32, fps: u32) -> (Vec<QuadVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idxs = Vec::new();

    // Total sheet width = sum of column widths; height = frame_count × ROW_HEIGHT.
    let total_w: f32 = XSheetPipeline::COL_WIDTHS.iter().sum();
    let total_h = (frame_count as f32) * XSheetPipeline::ROW_HEIGHT;
    let origin_x = -total_w * 0.5;
    let origin_y = -total_h * 0.5;

    // Row bands.
    for row in 0..frame_count {
        let y0 = origin_y + (row as f32) * XSheetPipeline::ROW_HEIGHT;
        let y1 = y0 + XSheetPipeline::ROW_HEIGHT;
        let is_sec = (row % fps) == 0;
        let shade = if is_sec {
            [0.86, 0.82, 0.70, 1.0] // 1-second accent row (slightly darker cream)
        } else if row % 2 == 0 {
            [0.96, 0.94, 0.86, 1.0]
        } else {
            [0.94, 0.92, 0.84, 1.0]
        };
        push_quad(&mut verts, &mut idxs, origin_x, y0, origin_x + total_w, y1, shade);
    }

    // Column dividers (1px ≈ 0.02 world units at default extent).
    let divider = [0.55, 0.50, 0.42, 1.0];
    let divider_w = 0.02;
    let mut cx = origin_x;
    for w in XSheetPipeline::COL_WIDTHS.iter() {
        cx += *w;
        push_quad(&mut verts, &mut idxs, cx - divider_w * 0.5, origin_y, cx + divider_w * 0.5, origin_y + total_h, divider);
    }

    // Outer border.
    let border = [0.35, 0.30, 0.25, 1.0];
    let bw = 0.04;
    push_quad(&mut verts, &mut idxs, origin_x - bw, origin_y - bw, origin_x, origin_y + total_h + bw, border);
    push_quad(&mut verts, &mut idxs, origin_x + total_w, origin_y - bw, origin_x + total_w + bw, origin_y + total_h + bw, border);
    push_quad(&mut verts, &mut idxs, origin_x, origin_y - bw, origin_x + total_w, origin_y, border);
    push_quad(&mut verts, &mut idxs, origin_x, origin_y + total_h, origin_x + total_w, origin_y + total_h + bw, border);

    (verts, idxs)
}

fn push_quad(verts: &mut Vec<QuadVertex>, idxs: &mut Vec<u32>, x0: f32, y0: f32, x1: f32, y1: f32, rgba: [f32; 4]) {
    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        QuadVertex { pos: [x0, y0], uv: [0.0, 1.0], rgba },
        QuadVertex { pos: [x1, y0], uv: [1.0, 1.0], rgba },
        QuadVertex { pos: [x1, y1], uv: [1.0, 0.0], rgba },
        QuadVertex { pos: [x0, y1], uv: [0.0, 0.0], rgba },
    ]);
    idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}
