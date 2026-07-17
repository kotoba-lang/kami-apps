# kami-apps — etzhayyim repo-specific (L3) product apps

The etzhayyim **`*.etzhayyim.com` product apps** — per-domain WASM front-ends
that consume the reusable engine. Per **ADR-2606011500**, the robotics/sim
apps (giemon / shibuya / tatekata / sarutahiko / funadaiku / …) are maintained
canonically **inside the `etzhayyim/kami-engine` submodule**, while this
monorepo workspace holds the product apps.

| Crate | Purpose | Site |
|---|---|---|
| `kami-app-bim` | BIM front-end | `bim.etzhayyim.com` |
| `kami-app-cad` | CAD front-end | `cad.etzhayyim.com` |
| `kami-app-live` | Live music venue app (consumes `kami-live` SDK) | `live.etzhayyim.com` |
| `kami-app-maps3d` | Nintendo-style 3D walkable map | `maps.etzhayyim.com` |
| `kami-app-animeka-timeline` | X-sheet + onion-skin timeline editor | animeka |

## Relationship to kami-engine

A **separate Cargo workspace** (`40-engine/kami-apps/`), sibling of the
`40-engine/kami-engine/` submodule. Each crate path-depends on the **L2**
engine + domain-lib crates under `../kami-engine/` (kami-app, kami-pipelines,
kami-render, kami-terrain, kami-vegetation, kami-bim, kami-cad, kami-live, …).
Run `git submodule update --init --recursive 40-engine/kami-engine` first.

## Build

```sh
cargo test --workspace                                    # native rlib + tests
wasm-pack build kami-app-maps3d --target web --release    # (run from this dir)
```

The deployed WASM bundles live in `60-apps/.../svelte/static/<app>/`; the
`.htm` pages load them from there, independent of crate-source location.
