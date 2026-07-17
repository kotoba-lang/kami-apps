//! kami-app-live — WASM entry for `live.etzhayyim.com`.
//!
//! Composes:
//! - `kami-pipelines::SkyAdapter`         — atmosphere
//! - `kami-pipelines::AtlasVisAdapter`    — beat-synced sprites (laser
//!                                          shock-waves, sparkles,
//!                                          fan light-sticks, flame pulses)
//! - `kami-pipelines::ParticleAdapter`    — confetti / drop bursts
//! with `kami-live::LiveShow` as the deterministic show clock.
//!
//! ```js
//! import init, { run_live_v1 } from './kami_app_live.js';
//! await init();
//! await run_live_v1('canvas');
//! ```
//!
//! The wasm-bindgen entry is a thin wiring layer; all show logic
//! (BPM, lighting, crowd, cheers) lives in `kami-live` and runs
//! identically on native (`cargo test -p kami-live`).

use kami_live::{
    AudioPattern, CheerKind, CrowdConfig, CueKind, CuePoint, Envelope, LightingCue,
    LightingFixture, LiveShow, StagePreset, Track, TrackId,
};

pub mod room_dto;
pub use room_dto::RoomConfig;

#[cfg(target_family = "wasm")]
use glam::Vec3;
#[cfg(target_family = "wasm")]
use kami_app::{CameraMode, InputMode, KamiApp, Position};
#[cfg(target_family = "wasm")]
use kami_live::{AudioCue, BeatEvent, ShowEvent};
#[cfg(target_family = "wasm")]
use log::Level;
#[cfg(target_family = "wasm")]
use std::cell::RefCell;
#[cfg(target_family = "wasm")]
use std::rc::Rc;

#[cfg(target_family = "wasm")]
use wasm_bindgen::JsValue;

/// JS bridges. The HTML shell defines `window.kamiPlayDrum`,
/// `window.kamiPlaySynth`, `window.kamiPlayPad`, `window.kamiSynthStop`.
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window, js_name = kamiPlayDrum, catch)]
    fn js_play_drum(slot: u32, velocity: f32) -> Result<(), JsValue>;

    #[wasm_bindgen(js_namespace = window, js_name = kamiPlaySynth, catch)]
    fn js_play_synth(midi: u32, velocity: f32, duration_seconds: f32) -> Result<(), JsValue>;

    #[wasm_bindgen(js_namespace = window, js_name = kamiPlayPad, catch)]
    fn js_play_pad(midi0: u32, midi1: u32, midi2: u32, midi3: u32, midi4: u32) -> Result<(), JsValue>;

    #[wasm_bindgen(js_namespace = window, js_name = kamiSynthStop, catch)]
    fn js_synth_stop() -> Result<(), JsValue>;
}

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

/// Build a default 3-track demo set. Replaces a network-sourced setlist
/// for the standalone demo at `live.etzhayyim.com/?demo=1`.
pub fn demo_show() -> LiveShow {
    let mut show = LiveShow::builder()
        .bpm(128.0)
        .stage(StagePreset::Hall)
        .crowd(CrowdConfig {
            fans_target: 600,
            cap: 4096,
            pit_bias: 0.65,
            seed: 7,
        })
        .performer_name("Mitama")
        .build();

    show.setlist_mut().push(Track {
        id: TrackId(1),
        title: "Opener (Wota Call)".into(),
        bpm: 128.0,
        length_beats: 128,
        cues: vec![
            CuePoint {
                at_beat: 32,
                kind: CueKind::Drop,
                tag: "first-drop".into(),
            },
            CuePoint {
                at_beat: 96,
                kind: CueKind::Drop,
                tag: "second-drop".into(),
            },
        ],
        dance: Some("wota".into()),
        audio: Some(AudioPattern::opener()),
    });
    show.setlist_mut().push(Track {
        id: TrackId(2),
        title: "Ballad Breakdown".into(),
        bpm: 92.0,
        length_beats: 96,
        cues: vec![CuePoint {
            at_beat: 16,
            kind: CueKind::Breakdown,
            tag: "sway".into(),
        }],
        dance: Some("hold".into()),
        audio: Some(AudioPattern::ballad()),
    });
    show.setlist_mut().push(Track {
        id: TrackId(3),
        title: "K-Pop Encore".into(),
        bpm: 140.0,
        length_beats: 128,
        cues: vec![
            CuePoint {
                at_beat: 16,
                kind: CueKind::Callout,
                tag: "hello-tokyo".into(),
            },
            CuePoint {
                at_beat: 64,
                kind: CueKind::Drop,
                tag: "encore-drop".into(),
            },
        ],
        dance: Some("kpop-point".into()),
        audio: Some(AudioPattern::encore()),
    });

    // Open with warm front-wash + a slow blue laser sweep.
    show.lighting_mut().push(
        LightingCue {
            fixture: LightingFixture::FrontPar,
            color: [1.0, 0.55, 0.35],
            intensity: 0.85,
            envelope: Envelope::Breathe,
            bars: 16,
        },
        0,
    );
    show.lighting_mut().push(
        LightingCue {
            fixture: LightingFixture::Laser,
            color: [0.2, 0.7, 1.0],
            intensity: 0.9,
            envelope: Envelope::Hold,
            bars: 24,
        },
        0,
    );
    show.lighting_mut().push(
        LightingCue {
            fixture: LightingFixture::Strobe,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            envelope: Envelope::Strobe { duty: 0.25 },
            bars: 4,
        },
        2, // strobe activates at bar 2 → after the intro
    );

    show.start();
    show
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub async fn run_live_v1(canvas_id: &str, room_json: Option<String>) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(Level::Info);
    log::info!("[live-v1] booting kami-app-live");

    // Audience-balcony perspective: 12 m up, 25 m back, looking toward stage.
    let spawn = Position::new(0.0, 4.0, -22.0);
    let app = KamiApp::new_web(canvas_id)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .with_label("live")
        .with_hud_publish(true)
        .with_camera(CameraMode::FirstPerson {
            spawn,
            yaw: 0.0,
            pitch: -0.05,
        })
        .with_input(InputMode::WasdFps);

    let sky = kami_pipelines::SkyAdapter::new(app.render_context()).with_overcast(0.05);
    let atlas = kami_pipelines::AtlasVisAdapter::new(app.render_context(), 4096);
    let particles = kami_pipelines::ParticleAdapter::new(app.render_context(), 4096);

    let initial = match room_json.as_deref() {
        Some(json) if !json.is_empty() => match serde_json::from_str::<RoomConfig>(json) {
            Ok(cfg) => match cfg.into_show() {
                Ok(s) => {
                    log::info!("[live-v1] booted from supplied room config");
                    s
                }
                Err(e) => {
                    log::warn!("[live-v1] room config build failed: {e}; falling back to demo");
                    demo_show()
                }
            },
            Err(e) => {
                log::warn!("[live-v1] room JSON parse failed: {e}; falling back to demo");
                demo_show()
            }
        },
        _ => demo_show(),
    };
    let show = Rc::new(RefCell::new(initial));
    // Publish the show to the thread-local slot so the JS bridge
    // (`live_send_cheer` / `live_set_room`) can act on the running show.
    SHOW_SLOT.with(|cell| {
        *cell.borrow_mut() = Some(show.clone());
    });
    let atlas_fx = atlas.clone();
    let particles_fx = particles.clone();
    let show_for_tick = show.clone();

    let app = app.on_update(move |_world, cam, dt| {
        let mut s = show_for_tick.borrow_mut();
        let events = s.tick(dt);
        let phase = s.grid().phase();

        // Beat-driven sparkles and shockwaves + audio bridge.
        for ev in events.iter() {
            match ev {
                ShowEvent::Audio(AudioCue::Drum { slot, velocity, .. }) => {
                    let _ = js_play_drum(*slot as u32, *velocity);
                }
                ShowEvent::Audio(AudioCue::Note {
                    midi,
                    velocity,
                    duration_beats,
                    ..
                }) => {
                    let secs = (*duration_beats * 60.0) / s.grid().bpm;
                    let _ = js_play_synth(*midi as u32, *velocity, secs);
                }
                ShowEvent::Audio(AudioCue::Pad { midis, .. }) => {
                    let _ = js_play_pad(
                        midis[0] as u32,
                        midis[1] as u32,
                        midis[2] as u32,
                        midis[3] as u32,
                        midis[4] as u32,
                    );
                }
                ShowEvent::Audio(AudioCue::Stop { .. }) => {
                    let _ = js_synth_stop();
                }
                ShowEvent::Beat(BeatEvent::Beat { .. }) => {
                    // SHOCK_WAVE in the centre on each kick.
                    atlas_fx.emit_pop(
                        Vec3::new(0.0, 1.4, 0.0),
                        kami_render::scene_pipelines::atlas_slot::SHOCK_WAVE,
                        [0.95, 0.45, 0.95],
                        2.5,
                        0.45,
                        0.0,
                    );
                }
                ShowEvent::Beat(BeatEvent::Bar { .. }) => {
                    // FLAME_LARGE puff on the bar — back-wash visual.
                    atlas_fx.emit_pop(
                        Vec3::new(0.0, 2.5, 1.5),
                        kami_render::scene_pipelines::atlas_slot::FLAME_LARGE,
                        [1.0, 0.7, 0.3],
                        1.6,
                        0.6,
                        0.0,
                    );
                }
                ShowEvent::Cue { cue, .. } if matches!(cue.kind, CueKind::Drop) => {
                    // Confetti burst out front.
                    particles_fx.burst(
                        Vec3::new(0.0, 4.5, -2.0),
                        140,
                        [1.0, 0.85, 0.4],
                    );
                    particles_fx.burst(
                        Vec3::new(-3.5, 4.5, -2.0),
                        80,
                        [0.95, 0.4, 0.7],
                    );
                    particles_fx.burst(
                        Vec3::new(3.5, 4.5, -2.0),
                        80,
                        [0.4, 0.7, 1.0],
                    );
                }
                _ => {}
            }
        }

        // Stage skeleton — emit a persistent sparkle at every fixture
        // mount + the four LED-wall corners. Cheap (~30 sprites/frame)
        // but gives the audience a visible architectural shell so the
        // "venue" reads as a venue and not just empty space.
        let stage = s.stage();
        for f in stage.fixtures.iter() {
            let tint = match f.fixture {
                LightingFixture::FrontPar => [1.0, 0.95, 0.7],
                LightingFixture::BackPar => [0.7, 0.85, 1.0],
                LightingFixture::Spot => [1.0, 1.0, 1.0],
                LightingFixture::Blinder => [1.0, 1.0, 0.85],
                LightingFixture::Strobe => [1.0, 1.0, 1.0],
                LightingFixture::Laser => [0.6, 1.0, 0.95],
            };
            atlas_fx.emit_static(
                f.position,
                kami_render::scene_pipelines::atlas_slot::SPARKLE_STAR,
                tint,
                0.32,
                0.10,
            );
        }
        // LED-wall corners.
        let w = stage.led_wall.half_size;
        let c = stage.led_wall.centre;
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            atlas_fx.emit_static(
                c + Vec3::new(sx * w.x, sy * w.y, 0.0),
                kami_render::scene_pipelines::atlas_slot::SHOCK_WAVE,
                [0.45, 0.7, 1.0],
                0.6,
                0.10,
            );
        }

        // Per-frame: emit fan light-sticks. Sample 24 fans per frame
        // (round-robin over the snapshot index) so we don't blow the
        // 4096-sprite atlas budget.
        let snapshot = s.snapshot();
        let stride = (snapshot.crowd.len() / 24).max(1);
        let frame_off = (cam.time * 4.0) as usize;
        for (i, fan) in snapshot.crowd.iter().enumerate() {
            if (i + frame_off) % stride != 0 {
                continue;
            }
            if !fan.stick_raised {
                continue;
            }
            atlas_fx.emit_static(
                fan.position + Vec3::new(0.0, 1.6, 0.0),
                kami_render::scene_pipelines::atlas_slot::SPARKLE_STAR,
                fan.stick_color,
                0.45,
                0.18,
            );
        }

        // Per-frame: emit laser arrow-trails along the resolved laser
        // direction. Cheap: one sprite per laser fixture.
        for frame in snapshot.lighting.iter() {
            if !matches!(frame.fixture, LightingFixture::Laser) {
                continue;
            }
            if frame.intensity < 0.05 {
                continue;
            }
            let origin = Vec3::new(0.0, 5.5, 0.5);
            for k in 1..6 {
                let t = k as f32 * 1.5;
                let p = origin + frame.aim * t;
                atlas_fx.emit_static(
                    p,
                    kami_render::scene_pipelines::atlas_slot::ARROW_TRAIL,
                    frame.color,
                    0.7,
                    0.12,
                );
            }
        }

        // Loudness HUD bridge — JS shell can read `__kami_hud_live.cheer`.
        let _ = phase;
    });

    log::info!("[live-v1] backend={:?}", app.backend());
    app.with_pipeline(sky)
        .with_pipeline(atlas)
        .with_pipeline(particles)
        .run()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// Thread-local holder for the active LiveShow. Wasm is single-threaded
// so a `thread_local!` is the simplest way to expose the running show
// to wasm-bindgen entry points called from JS (`live_send_cheer`).
#[cfg(target_family = "wasm")]
thread_local! {
    static SHOW_SLOT: RefCell<Option<Rc<RefCell<LiveShow>>>> = const { RefCell::new(None) };
}

fn parse_cheer_kind(s: &str) -> Option<CheerKind> {
    match s {
        "clap" => Some(CheerKind::Clap),
        "yell" => Some(CheerKind::Yell),
        "lightStick" | "light-stick" | "stick" => Some(CheerKind::LightStick),
        "jump" => Some(CheerKind::Jump),
        _ => None,
    }
}

/// External hook: ingest a cheer into the running show. The JS shell
/// (or a WebSocket relay) calls `live_send_cheer(kind, weight)`.
/// `kind` is one of "clap" | "yell" | "lightStick" | "jump".
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn live_send_cheer(kind: &str, weight: f32) {
    let Some(k) = parse_cheer_kind(kind) else {
        return;
    };
    let w = weight.max(0.0).min(10.0);
    SHOW_SLOT.with(|cell| {
        if let Some(show) = cell.borrow().as_ref() {
            show.borrow_mut().ingest_cheer(k, w);
        }
    });
}

/// Native (non-wasm) parser test surface — keeps `parse_cheer_kind`
/// reachable so the unit test below covers it.
#[cfg(not(target_family = "wasm"))]
pub fn parse_cheer_kind_for_test(s: &str) -> Option<CheerKind> {
    parse_cheer_kind(s)
}

/// Replace the running show with one built from `room_json`. Called by
/// the audience JS shell when the room DO broadcasts a `stateChange`
/// (the performer console pushed a new setlist).
///
/// Returns a JS error message on parse / build failure so the shell
/// can surface it (the renderer keeps running with the previous show).
#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn live_set_room(room_json: &str) -> Result<(), JsValue> {
    let cfg: RoomConfig = serde_json::from_str(room_json)
        .map_err(|e| JsValue::from_str(&format!("parse: {e}")))?;
    let new_show = cfg
        .into_show()
        .map_err(|e| JsValue::from_str(&format!("build: {e}")))?;
    SHOW_SLOT.with(|cell| {
        if let Some(rc) = cell.borrow().as_ref() {
            rc.replace(new_show);
        }
    });
    log::info!("[live-v1] applied room state from JS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kami_live::{BeatEvent, ShowEvent};

    #[test]
    fn cheer_kind_parser_accepts_known_strings() {
        for s in ["clap", "yell", "lightStick", "light-stick", "stick", "jump"] {
            assert!(
                parse_cheer_kind_for_test(s).is_some(),
                "should parse {s:?}"
            );
        }
        assert!(parse_cheer_kind_for_test("garbage").is_none());
        assert!(parse_cheer_kind_for_test("").is_none());
    }

    /// The native build of this crate must compose without GPU deps so
    /// CI smoke-tests can run it. Confirms the show wiring (cues fire,
    /// lighting frames produce, snapshot is non-empty).
    #[test]
    fn demo_show_drives_a_full_track() {
        let mut show = demo_show();
        let mut total_beats = 0;
        let mut drops = 0;
        for _ in 0..1200 {
            let evs = show.tick(1.0 / 30.0);
            for e in evs {
                match e {
                    ShowEvent::Beat(BeatEvent::Beat { .. }) => total_beats += 1,
                    ShowEvent::Cue { cue, .. } if matches!(cue.kind, CueKind::Drop) => drops += 1,
                    _ => {}
                }
            }
        }
        assert!(total_beats > 30);
        assert!(drops >= 1);
        let snap = show.snapshot();
        assert!(!snap.lighting.is_empty());
        assert!(!snap.crowd.is_empty());
    }
}
