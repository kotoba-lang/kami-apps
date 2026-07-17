//! Room config DTO. Mirrors the wire shape returned from the
//! `com.etzhayyim.apps.live.joinRoom` XRPC and pushed by the performer
//! console via `com.etzhayyim.apps.live.scheduleSet`. Lives in `kami-app-live`
//! (not `kami-live`) so the SDK's serde shape stays minimal.
//!
//! Conversion is one-way: `RoomConfig::into_show(...)` builds a fresh
//! `LiveShow`. The renderer's `RefCell<LiveShow>` swaps in the new show
//! without touching the wgpu pipelines.

use serde::{Deserialize, Serialize};

use kami_live::{
    AudioPattern, CrowdConfig, CueKind, CuePoint, Envelope, LightingCue, LightingFixture,
    LiveShow, StagePreset, Track, TrackId, VJDeck,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoomConfig {
    pub bpm: f32,
    /// Show start in unix-epoch seconds. Currently unused by the renderer
    /// (the local clock starts at the moment `into_show` is called) but
    /// kept on the wire for federation determinism.
    #[serde(default)]
    pub t0: f64,
    #[serde(default = "default_stage")]
    pub stage_preset: String,
    #[serde(default)]
    pub performer: Option<PerformerDto>,
    #[serde(default)]
    pub setlist: Vec<TrackDto>,
    #[serde(default)]
    pub lighting_program: Vec<LightingCueDto>,
    /// Optional crowd seed override. Default = 7.
    #[serde(default = "default_seed")]
    pub crowd_seed: u32,
    /// Fan target. Server clamps to stage preset if too high.
    #[serde(default = "default_fans")]
    pub fans_target: u32,
}

fn default_stage() -> String {
    "hall".to_string()
}

fn default_seed() -> u32 {
    7
}

fn default_fans() -> u32 {
    600
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerformerDto {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub vrm_cid: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackDto {
    pub id: u32,
    pub title: String,
    pub bpm: f32,
    pub length_beats: u32,
    #[serde(default)]
    pub dance: Option<String>,
    #[serde(default)]
    pub cues: Vec<CueDto>,
    /// Audio preset name: "opener" | "ballad" | "encore" | "none".
    #[serde(default)]
    pub audio: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CueDto {
    pub at_beat: u32,
    pub kind: String,
    #[serde(default)]
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LightingCueDto {
    pub fixture: String,
    pub color: [f32; 3],
    pub intensity: f32,
    pub envelope: String,
    /// Optional decay/duty parameters for `pulse` / `strobe` envelopes.
    #[serde(default)]
    pub envelope_param: Option<f32>,
    pub bars: u32,
    pub start_bar: u32,
}

impl RoomConfig {
    pub fn into_show(self) -> Result<LiveShow, String> {
        let stage = match self.stage_preset.as_str() {
            "club" => StagePreset::Club,
            "hall" => StagePreset::Hall,
            "festival" => StagePreset::Festival,
            other => return Err(format!("unknown stagePreset: {other:?}")),
        };
        let mut builder = LiveShow::builder()
            .bpm(self.bpm)
            .stage(stage)
            .crowd(CrowdConfig {
                fans_target: self.fans_target as usize,
                cap: 4096,
                pit_bias: 0.65,
                seed: self.crowd_seed,
            })
            .vj_deck(VJDeck::default_program());
        if let Some(p) = &self.performer {
            if !p.name.is_empty() {
                builder = builder.performer_name(&p.name);
            }
        }
        let mut show = builder.build();

        for t in self.setlist {
            let cues = t
                .cues
                .into_iter()
                .map(|c| {
                    let kind = match c.kind.as_str() {
                        "drop" => CueKind::Drop,
                        "breakdown" => CueKind::Breakdown,
                        "callout" => CueKind::Callout,
                        "custom" => CueKind::Custom,
                        _ => CueKind::Custom,
                    };
                    CuePoint {
                        at_beat: c.at_beat,
                        kind,
                        tag: c.tag,
                    }
                })
                .collect();
            let audio = match t.audio.as_deref() {
                Some("opener") => Some(AudioPattern::opener()),
                Some("ballad") => Some(AudioPattern::ballad()),
                Some("encore") => Some(AudioPattern::encore()),
                Some("none") | None => None,
                Some(_) => None,
            };
            show.setlist_mut().push(Track {
                id: TrackId(t.id),
                title: t.title,
                bpm: t.bpm,
                length_beats: t.length_beats,
                cues,
                dance: t.dance,
                audio,
            });
        }

        for c in self.lighting_program {
            let fixture = match c.fixture.as_str() {
                "frontPar" => LightingFixture::FrontPar,
                "backPar" => LightingFixture::BackPar,
                "spot" => LightingFixture::Spot,
                "blinder" => LightingFixture::Blinder,
                "laser" => LightingFixture::Laser,
                "strobe" => LightingFixture::Strobe,
                _ => continue,
            };
            let envelope = match c.envelope.as_str() {
                "hold" => Envelope::Hold,
                "pulse" => Envelope::Pulse {
                    decay: c.envelope_param.unwrap_or(5.0),
                },
                "breathe" => Envelope::Breathe,
                "strobe" => Envelope::Strobe {
                    duty: c.envelope_param.unwrap_or(0.25),
                },
                "ramp" => Envelope::Ramp,
                _ => Envelope::Hold,
            };
            show.lighting_mut().push(
                LightingCue {
                    fixture,
                    color: c.color,
                    intensity: c.intensity,
                    envelope,
                    bars: c.bars,
                },
                c.start_bar,
            );
        }

        show.start();
        Ok(show)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config_produces_running_show() {
        let json = r#"{
            "bpm": 120,
            "stagePreset": "club",
            "setlist": [{
                "id": 1,
                "title": "Test",
                "bpm": 120,
                "lengthBeats": 32,
                "dance": "wota",
                "audio": "opener",
                "cues": [{ "atBeat": 16, "kind": "drop", "tag": "test" }]
            }],
            "lightingProgram": [{
                "fixture": "frontPar",
                "color": [1.0, 0.5, 0.3],
                "intensity": 0.9,
                "envelope": "breathe",
                "bars": 8,
                "startBar": 0
            }]
        }"#;
        let cfg: RoomConfig = serde_json::from_str(json).expect("valid json");
        assert_eq!(cfg.setlist.len(), 1);
        let show = cfg.into_show().expect("show builds");
        assert_eq!(show.setlist().tracks.len(), 1);
        assert_eq!(show.setlist().tracks[0].title, "Test");
        assert!(show.setlist().tracks[0].audio.is_some());
    }

    #[test]
    fn unknown_stage_preset_errors() {
        let cfg = RoomConfig {
            bpm: 120.0,
            t0: 0.0,
            stage_preset: "stadium".into(),
            performer: None,
            setlist: vec![],
            lighting_program: vec![],
            crowd_seed: 1,
            fans_target: 10,
        };
        assert!(cfg.into_show().is_err());
    }

    #[test]
    fn unknown_envelope_falls_back_to_hold() {
        let cfg = RoomConfig {
            bpm: 120.0,
            t0: 0.0,
            stage_preset: "club".into(),
            performer: None,
            setlist: vec![],
            lighting_program: vec![LightingCueDto {
                fixture: "frontPar".into(),
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
                envelope: "weird".into(),
                envelope_param: None,
                bars: 4,
                start_bar: 0,
            }],
            crowd_seed: 1,
            fans_target: 10,
        };
        let show = cfg.into_show().expect("ok");
        let _ = show; // builds without panicking
    }

    #[test]
    fn camelcase_field_names_round_trip() {
        let original = RoomConfig {
            bpm: 128.0,
            t0: 1700000000.0,
            stage_preset: "hall".into(),
            performer: Some(PerformerDto { name: "Mitama".into(), vrm_cid: None }),
            setlist: vec![],
            lighting_program: vec![],
            crowd_seed: 42,
            fans_target: 200,
        };
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains("\"stagePreset\""));
        assert!(json.contains("\"crowdSeed\""));
        assert!(json.contains("\"fansTarget\""));
        let parsed: RoomConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.bpm, 128.0);
        assert_eq!(parsed.crowd_seed, 42);
    }
}
