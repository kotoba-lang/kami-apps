//! kami-app-cad — `cad.etzhayyim.com` per-game WASM entry.
//!
//! Phase 1: orbit camera + sky + a hardcoded mechanical part (base
//! plate + boss + central pin) composed from `kami-pipelines`
//! primitives (`unit_box`, `unit_cylinder`) via `CadSceneAdapter`.
//! Mouse-projected left-click picks the feature under the cursor and
//! highlights it amber; the pick is published to
//! `window.__kami_cad_pick` for the HUD.
//!
//! The real BREP → mesh path (via `kami-cad` + the `cad-job.etzhayyim.com`
//! Container tessellator) lands in the follow-up PR.

use glam::Mat4;
#[cfg(any(target_family = "wasm", test))]
use glam::Vec3;
#[cfg(any(target_family = "wasm", test))]
use kami_app::{CameraMode, InputMode};
#[cfg(target_family = "wasm")]
use kami_app::KamiApp;
#[cfg(target_family = "wasm")]
use kami_pipelines::{CadSceneAdapter, SkyAdapter};
use kami_pipelines::{unit_box, unit_cylinder};
use kami_render::RenderContext;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

/// JS entry: `import init, { run_cad_v2 } from './kami_app_cad.js';`
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub async fn run_cad_v2(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let app = KamiApp::new_web(canvas_id)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .with_label("cad")
        .with_hud_publish(true)
        .with_camera(CameraMode::Orbit {
            target: Vec3::new(0.0, 0.05, 0.0),
            distance: 0.5,
            yaw: 0.9,
            pitch: 0.45,
        })
        .with_input(InputMode::OrbitMouse);

    let sky = SkyAdapter::new(app.render_context());
    let cad = CadSceneAdapter::new(app.render_context());
    populate_demo_part(app.render_context(), &cad);
    log::info!("[cad-v2] batches={}", cad.batch_count());

    let cad_handle = cad.clone();

    let app = app
        .with_pipeline(sky)
        .with_pipeline(cad)
        .on_update(move |_world, camera, _dt| {
            if let Some(pick) = cad_handle.pick_from_camera_if_clicked(camera) {
                cad_handle.set_highlighted_by_id(&pick.feature_id);
                publish_pick(&pick);
                log::info!(
                    "[cad-v2] picked id={} name={:?} dist={:.3}",
                    pick.feature_id, pick.feature_name, pick.distance
                );
            }
        });
    log::info!("[cad-v2] backend={:?}", app.backend());
    app.run().await.map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Build the Phase-1 demo part directly into a `CadSceneAdapter`.
/// Shapes (in metres):
/// - 0.30 × 0.02 × 0.20 base plate  (feature `base`)
/// - 0.12 × 0.05 × 0.12 boss on top (feature `boss`)
/// - Ø 0.04 × 0.10 cylindrical pin  (feature `pin`)
pub fn populate_demo_part<A: CadSceneDriver>(ctx: &RenderContext, cad: &A) {
    let steel = [0.78, 0.80, 0.84];
    let boss = [0.35, 0.42, 0.52];
    let pin = [0.82, 0.55, 0.25];

    let (pp, pn, pi) = unit_box();
    cad.push_triangles(ctx, "base", "Base plate", &pp, &pn, &pi, steel,
        Mat4::from_scale_rotation_translation(
            glam::Vec3::new(0.30, 0.02, 0.20),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.01, 0.0),
        ));

    let (bp, bn, bi) = unit_box();
    cad.push_triangles(ctx, "boss", "Boss", &bp, &bn, &bi, boss,
        Mat4::from_scale_rotation_translation(
            glam::Vec3::new(0.12, 0.05, 0.12),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.045, 0.0),
        ));

    let (cp, cn, ci) = unit_cylinder(32);
    cad.push_triangles(ctx, "pin", "Pin (Ø4 mm × 100 mm)", &cp, &cn, &ci, pin,
        Mat4::from_scale_rotation_translation(
            glam::Vec3::new(0.04, 0.10, 0.04),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.07, 0.0),
        ));
}

/// Trait alias so the demo builder can drive either a real
/// `CadSceneAdapter` or a native counting stub in tests without
/// dragging in wgpu.
pub trait CadSceneDriver {
    fn push_triangles(
        &self,
        ctx: &RenderContext,
        feature_id: &str,
        feature_name: &str,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        base_color: [f32; 3],
        world: Mat4,
    );
}

#[cfg(target_family = "wasm")]
impl CadSceneDriver for kami_pipelines::CadSceneAdapter {
    fn push_triangles(
        &self,
        ctx: &RenderContext,
        feature_id: &str,
        feature_name: &str,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        base_color: [f32; 3],
        world: Mat4,
    ) {
        kami_pipelines::CadSceneAdapter::push_triangles(
            self, ctx, feature_id, feature_name, positions, normals, indices, base_color, world,
        );
    }
}

/// Publish the latest pick to `window.__kami_cad_pick` for the HUD.
#[cfg(target_family = "wasm")]
fn publish_pick(pick: &kami_pipelines::CadPick) {
    use wasm_bindgen::JsCast;
    let Some(window) = web_sys::window() else { return };
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"featureId".into(), &pick.feature_id.clone().into());
    let _ = js_sys::Reflect::set(&obj, &"featureName".into(), &pick.feature_name.clone().into());
    let _ = js_sys::Reflect::set(&obj, &"x".into(), &pick.world_pos.x.into());
    let _ = js_sys::Reflect::set(&obj, &"y".into(), &pick.world_pos.y.into());
    let _ = js_sys::Reflect::set(&obj, &"z".into(), &pick.world_pos.z.into());
    let _ = js_sys::Reflect::set(&obj, &"distance".into(), &pick.distance.into());
    let _ = js_sys::Reflect::set(&obj, &"at".into(), &js_sys::Date::now().into());
    let _ = js_sys::Reflect::set(&window.unchecked_into::<js_sys::Object>(), &"__kami_cad_pick".into(), &obj);
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    struct CountingCad {
        pushes: std::cell::Cell<usize>,
        total_indices: std::cell::Cell<usize>,
    }
    impl CadSceneDriver for CountingCad {
        fn push_triangles(
            &self,
            _ctx: &RenderContext,
            _feature_id: &str,
            _feature_name: &str,
            _positions: &[[f32; 3]],
            _normals: &[[f32; 3]],
            indices: &[u32],
            _base_color: [f32; 3],
            _world: Mat4,
        ) {
            self.pushes.set(self.pushes.get() + 1);
            self.total_indices.set(self.total_indices.get() + indices.len());
        }
    }

    #[test]
    fn builder_compiles() {
        let _ = CameraMode::Orbit {
            target: Vec3::ZERO,
            distance: 0.5,
            yaw: 0.0,
            pitch: 0.0,
        };
        let _ = InputMode::OrbitMouse;
    }

    #[test]
    fn demo_part_shapes() {
        let (pp, _, pi) = unit_box();
        assert_eq!(pp.len(), 24);
        assert_eq!(pi.len(), 36);
        let (cp, _, ci) = unit_cylinder(32);
        assert_eq!(cp.len(), 4 * 32 + 1 + 2 * 32 + 1 + 2 * 32);
        assert_eq!(ci.len(), 192 + 96 + 96);
        let _ = CountingCad { pushes: std::cell::Cell::new(0), total_indices: std::cell::Cell::new(0) };
    }
}
