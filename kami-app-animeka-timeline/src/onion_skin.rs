//! Onion-skin pipeline.
//!
//! Phase 1: alpha-blended quad stack for prev / next / current frame at the
//! **left edge** of the X-sheet (preview column). The actual frame textures
//! are not yet wired — this draws 3 placeholder tinted quads to prove the
//! compositing stack works with alpha blending.
//!
//! Phase 2 wiring:
//!   - Accept `Vec<wgpu::Texture>` indexed by frame number (loaded from
//!     blob CIDs via `com.etzhayyim.animeka.getCut` → `keyframe.imageCid` /
//!     `inbetween.imageCid`).
//!   - Bind prev (α=0.30, blue tint) + next (α=0.30, red tint) + current (α=1.0).
//!   - Scrub via playhead `u32` frame index passed through uniforms.

use bytemuck::{Pod, Zeroable};
use hecs::World;
use kami_app::{Camera, GpuCtx, RenderPipeline};

use crate::QuadVertex;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct OnionUniforms {
    view_proj: [[f32; 4]; 4],
}

const WGSL: &str = r#"
struct Uniforms { view_proj: mat4x4<f32> };
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

pub struct OnionSkinPipeline {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl OnionSkinPipeline {
    /// Preview column position relative to origin (left side of X-sheet).
    /// Width = ROW_HEIGHT × 3.2, height = ROW_HEIGHT × 3.2 (square preview).
    const PREVIEW_CX: f32 = -6.8;
    const PREVIEW_CY: f32 = 0.0;
    const PREVIEW_HALF: f32 = 1.6;

    pub fn new(ctx: &GpuCtx) -> Self {
        let device = &ctx.device;

        // 3 stacked quads: prev (blue, α=0.30), next (red, α=0.30), current (white, α=1.0).
        let mut verts = Vec::new();
        let mut idxs = Vec::new();
        push_quad(&mut verts, &mut idxs, Self::PREVIEW_CX - Self::PREVIEW_HALF, Self::PREVIEW_CY - Self::PREVIEW_HALF, Self::PREVIEW_CX + Self::PREVIEW_HALF, Self::PREVIEW_CY + Self::PREVIEW_HALF, [0.45, 0.62, 0.88, 0.30]);
        push_quad(&mut verts, &mut idxs, Self::PREVIEW_CX - Self::PREVIEW_HALF, Self::PREVIEW_CY - Self::PREVIEW_HALF, Self::PREVIEW_CX + Self::PREVIEW_HALF, Self::PREVIEW_CY + Self::PREVIEW_HALF, [0.88, 0.45, 0.45, 0.30]);
        push_quad(&mut verts, &mut idxs, Self::PREVIEW_CX - Self::PREVIEW_HALF, Self::PREVIEW_CY - Self::PREVIEW_HALF, Self::PREVIEW_CX + Self::PREVIEW_HALF, Self::PREVIEW_CY + Self::PREVIEW_HALF, [1.0, 1.0, 1.0, 1.0]);
        // Border around preview.
        let border = [0.35, 0.30, 0.25, 1.0];
        let bw = 0.04;
        push_quad(&mut verts, &mut idxs, Self::PREVIEW_CX - Self::PREVIEW_HALF - bw, Self::PREVIEW_CY - Self::PREVIEW_HALF - bw, Self::PREVIEW_CX - Self::PREVIEW_HALF, Self::PREVIEW_CY + Self::PREVIEW_HALF + bw, border);
        push_quad(&mut verts, &mut idxs, Self::PREVIEW_CX + Self::PREVIEW_HALF, Self::PREVIEW_CY - Self::PREVIEW_HALF - bw, Self::PREVIEW_CX + Self::PREVIEW_HALF + bw, Self::PREVIEW_CY + Self::PREVIEW_HALF + bw, border);
        push_quad(&mut verts, &mut idxs, Self::PREVIEW_CX - Self::PREVIEW_HALF, Self::PREVIEW_CY - Self::PREVIEW_HALF - bw, Self::PREVIEW_CX + Self::PREVIEW_HALF, Self::PREVIEW_CY - Self::PREVIEW_HALF, border);
        push_quad(&mut verts, &mut idxs, Self::PREVIEW_CX - Self::PREVIEW_HALF, Self::PREVIEW_CY + Self::PREVIEW_HALF, Self::PREVIEW_CX + Self::PREVIEW_HALF, Self::PREVIEW_CY + Self::PREVIEW_HALF + bw, border);

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion-vtx"),
            size: (verts.len() * std::mem::size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&verts));

        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion-idx"),
            size: (idxs.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&index_buf, 0, bytemuck::cast_slice(&idxs));

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion-uni"),
            size: std::mem::size_of::<OnionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onion-bgl"),
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
            label: Some("onion-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("onion-wgsl"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("onion-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("onion-rp"),
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

        Self { vertex_buf, index_buf, index_count: idxs.len() as u32, uniform_buf, bind_group, pipeline }
    }
}

impl RenderPipeline for OnionSkinPipeline {
    fn prepare(&mut self, ctx: &GpuCtx, camera: &Camera, _world: &World) {
        let uni = OnionUniforms { view_proj: camera.view_projection().to_cols_array_2d() };
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
            label: Some("onion-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
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
