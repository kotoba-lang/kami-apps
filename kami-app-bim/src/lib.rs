//! kami-app-bim — `bim.etzhayyim.com` per-game WASM entry.
//!
//! Phase 1: orbit camera + sky + a hardcoded 12 m × 8 m office storey
//! (1 slab, 1 ceiling, 4 exterior walls, 3 interior partitions, 1
//! column). The storey is constructed as a `kami_bim::StoreyScene` so
//! the same code path exercises the IFC-native types that the real
//! `com.etzhayyim.apps.bim.getStoreyScene` XRPC response will return.
//!
//! Entry naming follows ARCHITECTURE.md §Migration: `run_*_v2`.

use glam::{DAffine3, DMat3, DVec3};
use kami_bim::{
    BimId, ElementKind, Highlight, SceneGeom, SceneItem, StoreyScene,
};
use kami_pipelines::unit_box;

#[cfg(any(target_family = "wasm", test))]
use glam::Vec3;
#[cfg(any(target_family = "wasm", test))]
use kami_app::{CameraMode, InputMode};
#[cfg(target_family = "wasm")]
use kami_app::KamiApp;
#[cfg(target_family = "wasm")]
use kami_pipelines::{BimSceneAdapter, SkyAdapter};

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

/// JS entry (demo fallback): `import init, { run_bim_v2 } from './kami_app_bim.js';`
/// Renders the hardcoded `demo_office_storey()` — used when no XRPC
/// `getStoreyScene` response is available.
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub async fn run_bim_v2(canvas_id: &str) -> Result<(), JsValue> {
    let storey = demo_office_storey();
    let json = serde_json::to_string(&storey)
        .map_err(|e| JsValue::from_str(&format!("demo serialize: {e}")))?;
    run_bim_v2_with_scene(canvas_id, &json).await
}

/// JS entry (XRPC-fed): parses a JSON string produced by
/// `com.etzhayyim.apps.bim.getStoreyScene` into a `kami_bim::StoreyScene`
/// and feeds it to the `BimSceneAdapter`. Mouse-projected click =
/// pick + amber highlight + JS bridge publish.
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub async fn run_bim_v2_with_scene(canvas_id: &str, scene_json: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let storey: kami_bim::StoreyScene = serde_json::from_str(scene_json)
        .map_err(|e| JsValue::from_str(&format!("storey parse: {e}")))?;

    let centre = (storey.bounds_min + storey.bounds_max) * 0.5;
    let diag = (storey.bounds_max - storey.bounds_min).length() as f32;
    let distance = (diag * 1.25).max(12.0);

    let app = KamiApp::new_web(canvas_id)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .with_label("bim")
        .with_hud_publish(true)
        .with_camera(CameraMode::Orbit {
            target: Vec3::new(centre.x as f32, centre.y as f32, centre.z as f32),
            distance,
            yaw: 0.7,
            pitch: 0.45,
        })
        .with_input(InputMode::OrbitMouse);

    let sky = SkyAdapter::new(app.render_context());
    let bim = BimSceneAdapter::from_storey_scene(app.render_context(), &storey);
    log::info!(
        "[bim-v2] storey={:?} batches={}",
        storey.storey_name, bim.batch_count()
    );

    let bim_handle = bim.clone();

    let app = app
        .with_pipeline(sky)
        .with_pipeline(bim)
        .on_update(move |_world, camera, _dt| {
            if let Some(pick) = bim_handle.pick_from_camera_if_clicked(camera) {
                bim_handle.set_highlighted_by_id(&pick.element_id);
                publish_pick(&pick);
                log::info!(
                    "[bim-v2] picked id={} kind={:?} dist={:.2}",
                    pick.element_id, pick.kind, pick.distance
                );
            }
        });
    log::info!("[bim-v2] backend={:?}", app.backend());
    app.run().await.map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Publish the latest pick to `window.__kami_bim_pick` for the HUD.
#[cfg(target_family = "wasm")]
fn publish_pick(pick: &kami_pipelines::Pick) {
    use wasm_bindgen::JsCast;
    let Some(window) = web_sys::window() else { return };
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"elementId".into(), &pick.element_id.clone().into());
    let _ = js_sys::Reflect::set(&obj, &"kind".into(), &format!("{:?}", pick.kind).into());
    let _ = js_sys::Reflect::set(&obj, &"x".into(), &pick.world_pos.x.into());
    let _ = js_sys::Reflect::set(&obj, &"y".into(), &pick.world_pos.y.into());
    let _ = js_sys::Reflect::set(&obj, &"z".into(), &pick.world_pos.z.into());
    let _ = js_sys::Reflect::set(&obj, &"distance".into(), &pick.distance.into());
    let _ = js_sys::Reflect::set(&obj, &"at".into(), &js_sys::Date::now().into());
    let _ = js_sys::Reflect::set(&window.unchecked_into::<js_sys::Object>(), &"__kami_bim_pick".into(), &obj);
}

// ── Demo storey builder ────────────────────────────────────────────────

/// Build a minimal office storey: 12 m × 8 m × 3 m.
pub fn demo_office_storey() -> StoreyScene {
    const W: f64 = 12.0; // X extent
    const D: f64 = 8.0;  // Z extent
    const H: f64 = 3.0;  // ceiling height
    const T: f64 = 0.2;  // wall/slab thickness

    let mut items: Vec<SceneItem> = Vec::new();
    let mut id: u64 = 0;
    let mut push = |items: &mut Vec<SceneItem>, kind: ElementKind,
                    world: DAffine3, color: [f32; 3]| {
        id += 1;
        let (p, n, i) = unit_box_model_space();
        items.push(SceneItem {
            element_id: BimId::local(id),
            kind,
            world_transform: world,
            geom: SceneGeom::Triangles { positions: p, indices: i, normals: n },
            base_color: color,
            highlight: Highlight::None,
        });
    };

    // Slab (floor).
    push(&mut items, ElementKind::Slab,
        scale_translate(DVec3::new(W, T, D), DVec3::new(W * 0.5, -T * 0.5, D * 0.5)),
        [0.72, 0.70, 0.66]);

    // Ceiling slab.
    push(&mut items, ElementKind::Slab,
        scale_translate(DVec3::new(W, T, D), DVec3::new(W * 0.5, H + T * 0.5, D * 0.5)),
        [0.86, 0.85, 0.82]);

    // Exterior walls (4).
    // North (−Z face, at z=0).
    push(&mut items, ElementKind::Wall,
        scale_translate(DVec3::new(W, H, T), DVec3::new(W * 0.5, H * 0.5, T * 0.5)),
        [0.82, 0.78, 0.70]);
    // South (+Z face, at z=D).
    push(&mut items, ElementKind::Wall,
        scale_translate(DVec3::new(W, H, T), DVec3::new(W * 0.5, H * 0.5, D - T * 0.5)),
        [0.82, 0.78, 0.70]);
    // West (−X face, at x=0).
    push(&mut items, ElementKind::Wall,
        scale_translate(DVec3::new(T, H, D), DVec3::new(T * 0.5, H * 0.5, D * 0.5)),
        [0.82, 0.78, 0.70]);
    // East (+X face, at x=W).
    push(&mut items, ElementKind::Wall,
        scale_translate(DVec3::new(T, H, D), DVec3::new(W - T * 0.5, H * 0.5, D * 0.5)),
        [0.82, 0.78, 0.70]);

    // Interior partitions (3), splitting the floor into a corridor +
    // 2 meeting rooms.
    // Partition A: runs along X at z=3, from x=0..5.
    push(&mut items, ElementKind::Wall,
        scale_translate(DVec3::new(5.0, H, T), DVec3::new(2.5, H * 0.5, 3.0)),
        [0.90, 0.88, 0.82]);
    // Partition B: runs along X at z=3, from x=7..12.
    push(&mut items, ElementKind::Wall,
        scale_translate(DVec3::new(5.0, H, T), DVec3::new(9.5, H * 0.5, 3.0)),
        [0.90, 0.88, 0.82]);
    // Partition C: runs along Z from corridor into room, x=5 .. x=5.2.
    push(&mut items, ElementKind::Wall,
        scale_translate(DVec3::new(T, H, 2.0), DVec3::new(5.0, H * 0.5, 2.0)),
        [0.90, 0.88, 0.82]);

    // Round column (square-proxy for Phase 1) in the centre of the
    // south room.
    push(&mut items, ElementKind::Column,
        scale_translate(DVec3::new(0.45, H, 0.45), DVec3::new(6.0, H * 0.5, 6.0)),
        [0.55, 0.55, 0.58]);

    StoreyScene {
        storey_id: BimId::local(0),
        storey_name: "Demo GF".into(),
        elevation: 0.0,
        items,
        bounds_min: DVec3::new(0.0, -T, 0.0),
        bounds_max: DVec3::new(W, H + T, D),
    }
}

fn unit_box_model_space() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    unit_box()
}

/// Build a `DAffine3` that scales the unit box to `size` and places its
/// centre at `centre`. The unit box spans −0.5..+0.5 on each axis.
fn scale_translate(size: DVec3, centre: DVec3) -> DAffine3 {
    DAffine3 {
        matrix3: DMat3::from_cols(
            DVec3::new(size.x, 0.0, 0.0),
            DVec3::new(0.0, size.y, 0.0),
            DVec3::new(0.0, 0.0, size.z),
        ),
        translation: centre,
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    #[test]
    fn demo_storey_has_expected_batches() {
        let s = demo_office_storey();
        // 2 slabs + 4 exterior walls + 3 partitions + 1 column = 10
        assert_eq!(s.items.len(), 10);
        assert!(s.bounds_max.x > s.bounds_min.x);
    }

    #[test]
    fn builder_compiles() {
        let _ = CameraMode::Orbit {
            target: Vec3::ZERO,
            distance: 22.0,
            yaw: 0.0,
            pitch: 0.0,
        };
        let _ = InputMode::OrbitMouse;
    }
}
