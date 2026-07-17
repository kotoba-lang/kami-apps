//! kami-app-maps3d — Nintendo-style 3D walkable map for `maps.etzhayyim.com`.
//!
//! Phase 1 scope (this file):
//!   - WebGPU FPS walker (gravity + jump) atop procedural Plains terrain.
//!   - Sky + Water shared from `kami-pipelines`.
//!   - Building extrude pipeline (`buildings.rs`) consuming JSON injected
//!     from JS via `set_buildings_json` (data origin: maps-ui XRPC
//!     `com.etzhayyim.apps.maps.getChunk` — see `maps-3d.htm` shell).
//!   - Buildings double as wall colliders + walkable rooftops.
//!
//! Out of scope (Phase 2+):
//!   - H3-cell streaming auto-fetch from inside WASM.
//!   - Earcut of arbitrary polygon rings (today: AABB-of-footprint).
//!   - DEM displacement of procedural ground (today: pure FBM).
//!   - Per-building actor link-back overlay.
//!
//! ```js
//! import init, {
//!   run_maps3d, set_buildings_json,
//!   set_vegetation_json, set_atoms_json
//! } from './kami_app_maps3d.js';
//! await init();
//! await run_maps3d('gc');
//! // After a getChunk fetch:
//! set_buildings_json(JSON.stringify(boxes));
//! set_vegetation_json(JSON.stringify(vegItems));
//! set_atoms_json(JSON.stringify(atomItems));
//! ```

#[cfg(target_family = "wasm")]
use kami_app::{CameraMode, InputMode, KamiApp, Position};
#[cfg(target_family = "wasm")]
use log::Level;
#[cfg(target_family = "wasm")]
use std::cell::RefCell;
#[cfg(target_family = "wasm")]
use std::rc::Rc;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

pub mod atoms;
pub mod buildings;
pub mod gsplat;
pub mod mesh_tiles;
pub mod vegetation;
pub use atoms::AtomAdapter;
pub use buildings::{BuildingBox, BuildingExtrudeAdapter};
pub use gsplat::{parse_format as parse_gsplat_format, GsplatAdapter, GsplatFormat};
pub use mesh_tiles::MeshTileAdapter;
pub use vegetation::VegetationAdapter;

/// Plains-biome heightmap config used for both render and floor probe.
/// Returned cached so the floor probe samples the same FBM that
/// `TerrainAdapter::streaming` regenerates per-chunk.
fn plains_heightmap_config() -> kami_terrain::HeightmapConfig {
    kami_terrain::BiomePreset::Plains.heightmap(MAPS3D_TERRAIN_SEED)
}

const MAPS3D_TERRAIN_SEED: f32 = 137.0;
#[cfg(target_family = "wasm")]
const MAPS3D_CHUNK_EXTENT: u32 = 128;
#[cfg(target_family = "wasm")]
const MAPS3D_VIEW_RADIUS: i32 = 2;

/// Sample procedural ground height at world XZ. Mirrors what
/// `kami_terrain::Heightmap::generate` would emit at that exact world
/// position; safe to call every frame from a floor-probe closure.
fn sample_terrain_height(world_x: f32, world_z: f32) -> f32 {
    // 1×1 sample-tile so we don't allocate a multi-cell heightmap per
    // call. `Heightmap::generate` interpolates from integer lattice
    // anyway, so a 2×2 tile centred on (floor(x), floor(z)) gives a
    // bilinearly-correct value at fractional offsets.
    let cfg = plains_heightmap_config();
    let ix = world_x.floor();
    let iz = world_z.floor();
    let hm = kami_terrain::Heightmap::generate(2, 2, ix, iz, &cfg);
    hm.sample(world_x - ix, world_z - iz)
}

#[cfg(target_family = "wasm")]
thread_local! {
    /// Latest building set queued by JS. Re-read each frame by the
    /// floor probe + collider closures. The render adapter reads from
    /// its own `Rc<Shared>`; the queue here mirrors the same data so
    /// all probes (called outside the adapter) see the same set.
    static BUILDINGS_HANDLE: RefCell<Option<Rc<RefCell<Vec<BuildingBox>>>>> = RefCell::new(None);
    /// Render adapter handle — the wasm-bindgen `set_buildings_json`
    /// extern needs to push into the adapter from outside `run_maps3d`.
    static ADAPTER_HANDLE: RefCell<Option<BuildingExtrudeAdapter>> = RefCell::new(None);
    /// Photogrammetry tile adapter — `set_mesh_tile` upserts GLB bytes
    /// into this from outside `run_maps3d`.
    static MESH_HANDLE: RefCell<Option<MeshTileAdapter>> = RefCell::new(None);
    /// DB-driven vegetation adapter — `set_vegetation_json` pushes
    /// positioned TaxonomicProfile instances.
    static VEGETATION_HANDLE: RefCell<Option<VegetationAdapter>> = RefCell::new(None);
    /// CPK atom sphere adapter — `set_atoms_json` pushes element
    /// positions from `vertex_periodic_element` data.
    static ATOM_HANDLE: RefCell<Option<AtomAdapter>> = RefCell::new(None);
    /// Gaussian splat preview adapter — `set_gsplat_asset` upserts
    /// PLY / .splat tiles for landmark / spot QC review.
    static GSPLAT_HANDLE: RefCell<Option<GsplatAdapter>> = RefCell::new(None);
}

/// Push a fresh building set from JS. Returns parse error as a string
/// so the caller can surface it. See `buildings.rs::BuildingBoxJson`
/// for the expected JSON shape.
///
/// ```js
/// set_buildings_json(JSON.stringify([
///   { minX: -10, maxX: 10, minZ: -10, maxZ: 10, baseY: 0, height: 24 }
/// ]));
/// ```
/// Upsert a photogrammetry tile from GLB bytes. Called by the JS host
/// after fetching `b2://etzhayyim-nats/maps3d/tile/{tile_h3}.glb` (output
/// of the `maps3d.simplifyAndExport` BPMN task). Replaces any prior
/// mesh for the same tile.
///
/// ```js
/// const glb = await fetch(meshUri).then(r => r.arrayBuffer());
/// set_mesh_tile(tileH3, new Uint8Array(glb));
/// ```
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn set_mesh_tile(tile_h3: &str, glb: &[u8]) -> Result<(), JsValue> {
    let mut handled = false;
    MESH_HANDLE.with(|h| {
        if let Some(adapter) = h.borrow().as_ref() {
            adapter
                .upsert_tile(tile_h3, glb)
                .map_err(|e| JsValue::from_str(&format!("mesh tile {tile_h3}: {e}")))
                .map(|_| handled = true)
                .ok();
        }
    });
    if !handled {
        return Err(JsValue::from_str(
            "mesh adapter not initialised (call run_maps3d first)",
        ));
    }
    Ok(())
}

/// Drop a previously-loaded photogrammetry tile so its GPU buffers can
/// be released. Used by the JS host's tile cache eviction.
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn remove_mesh_tile(tile_h3: &str) {
    MESH_HANDLE.with(|h| {
        if let Some(adapter) = h.borrow().as_ref() {
            adapter.remove_tile(tile_h3);
        }
    });
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn set_buildings_json(json: &str) -> Result<(), JsValue> {
    let parsed: Vec<buildings::BuildingBoxJson> = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("buildings json: {e}")))?;
    let boxes: Vec<BuildingBox> = parsed.into_iter().map(Into::into).collect();
    BUILDINGS_HANDLE.with(|h| {
        if let Some(rc) = h.borrow().as_ref() {
            *rc.borrow_mut() = boxes.clone();
        }
    });
    ADAPTER_HANDLE.with(|h| {
        if let Some(adapter) = h.borrow().as_ref() {
            adapter.set_boxes(boxes);
        }
    });
    Ok(())
}

/// Push a fresh vegetation set from JS. Each item's `renderProfileJson`
/// encodes an `OwnedTaxonomicProfile` (camelCase, from the
/// `seibutsu.renderProfile` XRPC shape). World position + scale are
/// applied at build time so the GPU sees pre-transformed geometry.
///
/// ```js
/// set_vegetation_json(JSON.stringify([
///   { renderProfileJson: '{"canopy":"Blade",...}',
///     worldX: 5.0, worldY: 0.0, worldZ: -3.0,
///     scaleX: 1.0, scaleY: 1.2, scaleZ: 1.0 }
/// ]));
/// ```
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn set_vegetation_json(json: &str) -> Result<(), JsValue> {
    let items: Vec<vegetation::VegetationItemJson> = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("vegetation json: {e}")))?;
    VEGETATION_HANDLE.with(|h| {
        if let Some(adapter) = h.borrow().as_ref() {
            adapter.set_items(items);
        }
    });
    Ok(())
}

/// Push a fresh atom set from JS. Each item carries CPK colour + sphere
/// radius (pm) + world position. Radius is scaled `× 0.001` so a
/// 150 pm atom renders at ~0.15 m diameter in world space.
///
/// ```js
/// set_atoms_json(JSON.stringify([
///   { symbol:"C", colorR:0.2, colorG:0.2, colorB:0.2,
///     sphereRPm:77, worldX:0, worldY:1.5, worldZ:0 }
/// ]));
/// ```
/// Push a splat tile from JS. `format` is `"ply"` (default) or
/// `"splat"` (antimatter15 32-byte compact). Returns the parsed splat
/// count so the host UI can show a "loaded N splats" toast.
///
/// Capped at `kami_pipelines::MAX_SPLATS_PER_CLOUD = 100_000`
/// (preview/QC scope, ADR-2605092800). Use the bake pipeline for
/// heavier scenes.
///
/// ```js
/// const ply = await fetch(asset.signedUrl).then(r => r.arrayBuffer());
/// const n = set_gsplat_asset(asset.tileH3, new Uint8Array(ply), "ply");
/// console.log(`loaded ${n} splats for ${asset.tileH3}`);
/// ```
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn set_gsplat_asset(tile_h3: &str, bytes: &[u8], format: &str) -> Result<u32, JsValue> {
    let mut count = 0_u32;
    let mut handled = false;
    GSPLAT_HANDLE.with(|h| {
        if let Some(adapter) = h.borrow().as_ref() {
            match adapter.upsert_from_bytes(tile_h3, bytes, gsplat::parse_format(format)) {
                Ok(n) => {
                    count = n as u32;
                    handled = true;
                }
                Err(e) => {
                    handled = true;
                    log::warn!("[maps3d] gsplat upsert {tile_h3}: {e}");
                }
            }
        }
    });
    if !handled {
        return Err(JsValue::from_str(
            "gsplat adapter not initialised (call run_maps3d first)",
        ));
    }
    Ok(count)
}

/// Drop a previously-loaded splat tile so its GPU buffers can be
/// released. Used by the JS host's tile cache eviction.
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn remove_gsplat_asset(tile_h3: &str) {
    GSPLAT_HANDLE.with(|h| {
        if let Some(adapter) = h.borrow().as_ref() {
            adapter.remove(tile_h3);
        }
    });
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn set_atoms_json(json: &str) -> Result<(), JsValue> {
    let items: Vec<atoms::AtomItemJson> = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("atoms json: {e}")))?;
    ATOM_HANDLE.with(|h| {
        if let Some(adapter) = h.borrow().as_ref() {
            adapter.set_items(items);
        }
    });
    Ok(())
}

/// Boot the 3D map walker on the given canvas.
///
/// ```js
/// import init, { run_maps3d } from './kami_app_maps3d.js';
/// await init();
/// await run_maps3d('gc');
/// ```
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub async fn run_maps3d(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(Level::Info);
    log::info!("[maps3d] boot canvas={canvas_id}");

    // Spawn 2 m above the procedural ground at world origin. The floor
    // probe lifts the eye to `terrain + eye_height` on the first tick
    // anyway, so the exact spawn Y matters only as an upper bound.
    let spawn_xz = (0.0_f32, 0.0_f32);
    let spawn_y = sample_terrain_height(spawn_xz.0, spawn_xz.1) + 2.0;
    let spawn = Position::new(spawn_xz.0, spawn_y, spawn_xz.1);

    let app = KamiApp::new_web(canvas_id)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .with_label("maps3d")
        .with_hud_publish(true)
        .with_camera(CameraMode::FirstPerson {
            spawn,
            yaw: 0.0,
            pitch: -0.15,
        })
        .with_input(InputMode::WasdFps);

    // Shared pipelines.
    let sky = kami_pipelines::SkyAdapter::new(app.render_context());
    let terrain = kami_pipelines::TerrainAdapter::streaming(
        app.render_context(),
        kami_terrain::BiomePreset::Plains,
        MAPS3D_TERRAIN_SEED,
        MAPS3D_CHUNK_EXTENT,
        MAPS3D_VIEW_RADIUS,
    );
    // Sea level just below Plains sand_line — water fills inundated
    // valleys without flooding walkable ground.
    let water = kami_pipelines::WaterAdapter::new(app.render_context(), 1024.0, 1.0);

    // Building extrude pipeline (per-game). Two clones: one we register
    // with the app (consumed by `with_pipeline`), one stays in
    // `ADAPTER_HANDLE` so `set_buildings_json` can push fresh data
    // from JS later.
    let buildings = BuildingExtrudeAdapter::new(app.render_context());
    ADAPTER_HANDLE.with(|h| *h.borrow_mut() = Some(buildings.clone()));

    // Photogrammetry tile pipeline (render-only skin layered over the
    // OSM-extrude collision shapes). Tiles arrive via `set_mesh_tile`.
    let mesh_tiles = MeshTileAdapter::new(app.render_context());
    MESH_HANDLE.with(|h| *h.borrow_mut() = Some(mesh_tiles.clone()));

    // DB-driven vegetation — TaxonomicProfile instances at exact world
    // positions. Updated via `set_vegetation_json`.
    let vegetation = VegetationAdapter::new(app.render_context());
    VEGETATION_HANDLE.with(|h| *h.borrow_mut() = Some(vegetation.clone()));

    // CPK atom spheres — element positions from vertex_periodic_element.
    // Updated via `set_atoms_json`.
    let atoms = AtomAdapter::new(app.render_context());
    ATOM_HANDLE.with(|h| *h.borrow_mut() = Some(atoms.clone()));

    // Gaussian splat preview tiles — landmark / spot QC review only.
    // Tiles arrive via `set_gsplat_asset(tile_h3, bytes, format)` and
    // are dropped via `remove_gsplat_asset(tile_h3)`. Render order
    // intentionally last so splats over-blend onto the opaque mesh
    // tile / building / vegetation pass.
    let gsplat = GsplatAdapter::new(app.render_context());
    GSPLAT_HANDLE.with(|h| *h.borrow_mut() = Some(gsplat.clone()));

    // Mirror buffer that floor + collider probes read each frame. The
    // render adapter holds its own copy, but probes are independent
    // closures so we keep a parallel `Rc<RefCell<Vec<BuildingBox>>>`
    // they can borrow from outside the adapter.
    let probe_boxes: Rc<RefCell<Vec<BuildingBox>>> = Rc::new(RefCell::new(Vec::new()));
    BUILDINGS_HANDLE.with(|h| *h.borrow_mut() = Some(probe_boxes.clone()));

    let floor_probe_buildings = buildings.clone();
    let collider_probe_buildings = buildings;
    let collider_probe = move |min: glam::Vec3, max: glam::Vec3| -> bool {
        collider_probe_buildings.aabb_solid(min, max)
    };

    log::info!(
        "[maps3d] backend={:?} chunk_extent={MAPS3D_CHUNK_EXTENT} radius={MAPS3D_VIEW_RADIUS}",
        app.backend()
    );

    app.with_pipeline(sky)
        .with_pipeline(terrain)
        .with_pipeline(water)
        .with_pipeline(floor_probe_buildings.clone())
        .with_pipeline(mesh_tiles)
        .with_pipeline(vegetation)
        .with_pipeline(atoms)
        .with_pipeline(gsplat)
        // Floor = max(terrain, rooftop_under_player). Lets the player
        // walk on rooftops without falling through terrain when off any
        // building.
        .with_floor_probe(move |p| {
            let terrain_y = sample_terrain_height(p.x, p.z);
            let roof_y = floor_probe_buildings.rooftop_y(p);
            // Pick the higher surface at or below the eye. If a roof is
            // above the eye, ignore it (player is below the floor of
            // the building rather than on top).
            let candidates = [Some(terrain_y), roof_y]
                .into_iter()
                .flatten()
                .filter(|y| *y <= p.y + 0.01)
                .fold(None, |acc: Option<f32>, y| {
                    Some(acc.map_or(y, |a| a.max(y)))
                });
            candidates.or(Some(terrain_y))
        })
        .with_collider_probe(collider_probe)
        .with_eye_height(1.7)
        .with_player_radius(0.35)
        .with_gravity(18.0)
        .with_jump_impulse(7.0)
        .run()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── Native test stub: ensure non-wasm targets compile (cargo check) ──
#[cfg(not(target_family = "wasm"))]
pub fn run_maps3d() -> Result<(), String> {
    // Smoke check that helper math compiles + executes off-target.
    let _h = sample_terrain_height(0.0, 0.0);
    Ok(())
}
