//! kami-app-animeka-timeline — X-sheet + onion-skin timeline editor.
//!
//! Per-game crate for animeka.etzhayyim.com. Renders a 2D X-sheet grid (frame × layer
//! columns A/B/C + dialogue + camera), an onion-skin composite (prev/next frame
//! translucent + current frame opaque), and a playhead scrubber.
//!
//! Scaffold scope (Phase 1):
//!   - KamiApp bootstrap via `CameraMode::Ortho2D`
//!   - `XSheetPipeline` — draws the grid + frame numbers (placeholder text layout)
//!   - `OnionSkinPipeline` — alpha-blended quads for prev/next/current frame textures
//!
//! Follow-up (Phase 2):
//!   - Stylus pressure/tilt capture via `kami-input` (KAMI `FocusManager` routing)
//!   - Per-frame `wgpu::Texture` upload from blob CIDs
//!   - Playback scrubber with keyboard / mouse scroll
//!   - Integration with `com.etzhayyim.animeka.*` XRPC handlers (keyframe / inbetween record load)

#[cfg(target_family = "wasm")]
use glam::Vec3;
#[cfg(target_family = "wasm")]
use kami_app::{CameraMode, InputMode, KamiApp};
#[cfg(target_family = "wasm")]
use log::Level;

pub mod onion_skin;
pub mod xsheet;

pub use onion_skin::OnionSkinPipeline;
pub use xsheet::XSheetPipeline;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

/// Entry point exported to JS.
///
/// ```js
/// import init, { run_animeka_timeline } from './kami_app_animeka_timeline.js';
/// await init();
/// await run_animeka_timeline('gc');
/// ```
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub async fn run_animeka_timeline(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(Level::Info);
    log::info!("[animeka-timeline] starting");

    // X-sheet conventions: 24 fps default, 32 frames visible, 5 layer columns (A/B/C + dialogue + camera).
    let fps = 24_u32;
    let frame_count = 32_u32;
    let lanes = 5_u32;

    let app = KamiApp::new_web(canvas_id)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .with_label("animeka-timeline")
        .with_hud_publish(true)
        .with_camera(CameraMode::Ortho2D {
            center: Vec3::new(0.0, 0.0, 0.0),
            extent: (frame_count as f32) * XSheetPipeline::ROW_HEIGHT * 0.55,
        })
        .with_input(InputMode::None);

    let xsheet = XSheetPipeline::new(app.render_context(), frame_count, lanes, fps);
    let onion = OnionSkinPipeline::new(app.render_context());

    app.with_pipeline(xsheet)
        .with_pipeline(onion)
        .run()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Native-side smoke test (cargo test) — verifies adapters can be referenced without a surface.
#[cfg(not(target_family = "wasm"))]
pub fn smoke_test() -> Result<(), String> {
    log::info!("[animeka-timeline] native smoke test — no surface bound");
    Ok(())
}

/// Shared 2D-quad vertex layout used by both X-sheet grid cells and onion-skin
/// frame layers. Position is in ortho world space (y up), UV in `[0, 1]`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub rgba: [f32; 4],
}

impl QuadVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 2 },
        ],
    };
}
