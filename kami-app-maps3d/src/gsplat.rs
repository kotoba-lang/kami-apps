//! Splat asset thin wrapper for `kami-app-maps3d`.
//!
//! All rendering lives in `kami_pipelines::GsplatAdapter`. This module
//! exists only to keep the per-game wasm-bindgen surface focused on
//! map-tile semantics (`tile_h3` keys + `format` enum string).

pub use kami_pipelines::{GsplatAdapter, GsplatError, GsplatFormat, MAX_SPLATS_PER_CLOUD};

/// Map a JS-side format string ("ply" / "splat") to `GsplatFormat`.
/// Falls back to `Ply` for unknown values to keep host code resilient
/// (the parser will surface a clear error if the bytes really are not
/// a PLY).
pub fn parse_format(label: &str) -> GsplatFormat {
    match label.trim().to_ascii_lowercase().as_str() {
        "splat" | "compact" | "antimatter" => GsplatFormat::Splat,
        _ => GsplatFormat::Ply,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_format_default_ply() {
        assert!(matches!(parse_format("PLY"), GsplatFormat::Ply));
        assert!(matches!(parse_format(""), GsplatFormat::Ply));
        assert!(matches!(parse_format("foo"), GsplatFormat::Ply));
    }

    #[test]
    fn parse_format_splat_synonyms() {
        assert!(matches!(parse_format("splat"), GsplatFormat::Splat));
        assert!(matches!(parse_format("Compact"), GsplatFormat::Splat));
        assert!(matches!(parse_format("antimatter"), GsplatFormat::Splat));
    }
}
