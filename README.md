# dirtemu

Experimental **AMD RX 570 / Polaris** compatibility fork of
[ztrx](https://github.com/Zettrox/ztrx) for running the Xbox Series build of
GTA VI on Windows PCs with older AMD graphics cards.

**This repository contains no game files or assets.** You need your own copy
of the game. Not affiliated with or endorsed by its developer or publisher.

## Status (2026-11-27)

Work in progress. **Not a playable release yet.**

Target test machine:

- Windows 10 x64
- Intel Xeon E3-1240 v3 (4C/8T, AVX2)
- 16 GiB RAM
- AMD Radeon RX 570 8 GiB (Polaris, wave64, no mesh shaders, no DXR)
- System DirectX exposes SM 6.5; app-local Agility 1.620.2 exposes SM 6.7

What is already implemented and tested at shader/PSO level:

| Area | Status |
|---|---|
| SM 6.5 shader translation | implemented; the 22 baseline SM 6.6 GPU failures pass on RX 570 |
| SM 6.5 descriptor tables | implemented; alias-table PSO/dispatch tests pass on RX 570 |
| Agility / SM 6.6 | optional app-local runtime; real bindless and typed atomic64 GPU tests pass on RX 570; highest reported model is 6.7 |
| Wave32 compute groups | groups of at most 32 threads run inside physical wave64; 8/8 formerly refused PSOs create with the debug layer |
| Mesh shader fallback | compute emulation + vertex pulling + one `DrawInstanced`; 9 279 / 9 282 real mesh shaders translate |
| Amplification shaders | none found in the inspected shader package |
| BVH / RT profiles | `Full`, `StaticMiss`, `Limited(N)` implemented and tested on RX 570 |
| Shadowmask PSV fix | typed-UAV PSV ordering fixed; post/shadow validation improved to 97/104 |
| Ambient fallback | constant indirect-light fill after ScreenProbes temporal resolve, Polaris default 0.2 |
| BC3 texture fallback | five unaligned BC3 textures decoded to RGBA8 with original dimensions |
| Series-S memory profile | title memory, host-copy, texture and staging caps active |
| Barycentrics / `attributeAtVertex` | software geometry path when the device lacks barycentrics, including SM 6.6; ordinary triangles, rectangle lists and compute mesh fallback supported |
| Integer render targets | blending normalized so D3D12 accepts the pipeline |
| Conservative rasterization | ordinary raster coverage on unsupported devices; conservative edge coverage is not reproduced |
| Streamed textures | upload decisions follow staging refresh, so new data reaches the same submission; partial uploads retry by elapsed time |
| Memory pressure | reclaim unused staging buffers; configurable sampled-texture budget; root-window tails reduce duplicate data |
| Shared constants | optional direct GPU access to the title's constants section, with ordered indirect argument preparation |
| Precompilation | package shaders and every recorded pipeline state; persistent progress, bounded workers and translator-keyed disk cache |
| Full game boot / world / FPS | boot, menus and world image confirmed in earlier runs; new rendering fixes still need an in-game comparison after full precompilation |

Latest validation on the RX 570:

- **53/53 selected problem pipeline states** create successfully at both SM 6.5
  and Agility SM 6.6 with the D3D12 debug layer, with no refused or unreadable
  states. The second SM 6.6 preparation reuses all 53 from cache. Unsupported hardware
  barycentrics and conservative rasterization caused their earlier failures.
- GPU readback tests verify perspective and linear barycentric weights with
  unequal clip-space W, per-vertex attributes, and both halves of rectangle lists.
- A texture changed after command-list recording reaches the GPU in that same
  submission. Ten GPU regression checks and nine interpolation checks passed.
- The same installed driver reports SM 6.7 with Agility. A real SM 6.6 bindless
  compute shader performs 64 conflicting typed atomic64 operations correctly
  with dynamic texel coordinates; all returned original values match.

Known limitations:

- A Series S SM 6.6 run on 27 November lost the display signal after desktop
  artifacts; Windows recorded a video-engine timeout. Host-copy pressure and
  stale root-buffer binding faults were corrected, but the exact GPU fault
  and gameplay stability remain unverified. See [the incident analysis](docs/rx570-gpu-timeout-2026-11-27.md).
- The complete SM 6.6 archive pass prepared **25,586 translations**, refused
  303 records and skipped 614 duplicates out of 26,503 records. All 679 records
  previously refused for SM 6.6 image atomics now translate. Of the remaining
  refusals, 195 contain no stage supported by this translator; other gaps include
  descriptor control flow and descriptor data reads. Translation success does
  not by itself establish correct execution of every game shader.
- Driver 31.0.21925.1001 loses conflicting typed atomic updates for literal
  uniform texture coordinates in a synthetic test, including atomic32.
  Dynamic coordinates and explicit wave serialization pass. This is a separate
  correctness issue; an advertised shader model does not establish correctness
  for every shader. See [the capability investigation](docs/rx570-agility.md).
- Native wave32 compute groups larger than 32 threads still require subwave
  emulation on Polaris. They are excluded from the small-group wave64 correction;
  successful PSO creation alone does not verify their cross-lane semantics.
- The recorded set contains **83,562 pipeline states**. The first full warm
  with the final SM 6.6 translator must finish before the next game run.
  Preparation of every state is not yet a verified result.
- 16 GiB RAM is below the practical working set observed in related forks;
  a large pagefile is required and performance will still be limited.
- Shader translation and driver pipeline refusals still omit rendering passes.
- CPU preparation, demand paging and buffer read-back cause long frame times.
- Missing geometry, low-detail textures and watchdog failures in the previous
  game runs are not yet claimed to be fully fixed. No new gameplay FPS result
  is available for these changes.

## Requirements

- Windows 10/11 x64, a CPU with AVX2
- Direct3D 12 GPU
  - hardware path: shader model 6.6 + mesh shaders
  - RX 570 / Polaris path: shader model 6.5 or 6.6 + compute mesh fallback
- Rust (the version in `rust-toolchain.toml`)
- Several GiB of free Windows commit memory; for world loading a large
  pagefile is recommended (32 GiB or more on SSD)

## Build

```powershell
cargo build --release -p victory
```

To use the newer DirectX runtime on Windows/MSVC:

```powershell
cargo build --release -p victory --features agility
powershell -ExecutionPolicy Bypass -File scripts\setup-agility.ps1
```

The setup script downloads the official Microsoft Agility 1.620.2 package,
checks its pinned SHA256 and Microsoft signatures, and places SDK 620 in
`target\release\D3D12`. It changes no system DLLs or drivers. The `agility`
build requires those runtime files beside the executable. The default build
continues to use the system DirectX runtime.

Diagnostics build:

```powershell
cargo build --profile diag -p victory --features diagnostics
```

For diagnostics with Agility, use `--features diagnostics,agility` and run
the setup script with `-Destination target\diag`.

## Play

```powershell
powershell -ExecutionPolicy Bypass -File scripts\play.ps1 -Game "<your game folder>\<the game's exe>"
```

The launcher waits for shader translation and recorded D3D12 pipelines before
starting the title. The first run scans the package's shader archives and
DLLs. It records pipeline states when the title creates them. Later runs
prepare those exact states before the title starts. Keep the cache directory
on the same drive across runs. Pipeline recipes are specific to the game
build and console profile. Shader and driver refusals appear in the summary.

To warm the cache without starting the game, use the same console profile and
environment as the launcher.

```powershell
.\target\release\victory.exe precompile "<your game folder>\<the game's exe>" --console series-s --jobs 8
```

`--jobs` controls compilation threads. Without it, preparation uses every
logical CPU core, up to 16 threads; pass `--jobs 3` to reduce CPU and memory use.
Add `--used-only` to build only states used in previous runs.
For a bounded preparation, set `$env:VICTORY_PRECOMPILE_TOTAL = "20000"`
before running `precompile`. It selects at most the first 20,000 recipes,
prioritizing previously requested states, then sorting the remaining keys.
The limit includes checkpointed states and stays in force across workers and
resumed runs. Ready and refused states are reported separately. Remove the
variable to prepare the complete set.
The RX 570 launcher exposes the Series S console model to the game. Use the
same console model for preparation and play; recipes recorded as Series X
belong to a separate set.
Without that option, every recorded state is attempted. Use the same release
binary, cache, console profile and shader settings for preparation and play.
Failure details are stored in `shader-scan/*.failures.txt` in the configured
cache. Shader/pipeline preparation is separate from the game's asset streaming.
Translations are keyed by the translator sources, dependencies, target GPU
and settings. Rebuilding rendering or runtime code preserves them.
Completed pipeline preparation is checkpointed after saving the driver
library. Later runs skip those completed recipes; changing translator settings
or losing the compatible driver library invalidates the relevant progress.
Preparation runs in workers of at most 2,048 new states, so their temporary
memory is released between batches. Interrupted preparation resumes from the
last saved checkpoint. Within preparation workers, new driver pipelines are
kept until the next checkpoint and then stored together. This prevents
`StorePipeline` writes from blocking workers still loading and compiling
other states. The journal advances only after pending stores and the library
have been saved. Set `VICTORY_PRECOMPILE_DEFER_STORES=0` to use immediate stores.
Game rendering keeps immediate stores.
Refused states are counted separately and retried when
the preparation identity changes. Previously used pipeline objects are loaded
into the game process before execution.
Measurements and the unresolved compiler fault are recorded in
[the RX 570 preparation report](docs/rx570-precompile-2026-11-27.md).
Successful driver library loads can still be expensive on a first pass.
Set `VICTORY_PIPELINE_CACHE_TRACE=1` to distinguish loads from misses and time
both paths. The [follow-up report](docs/rx570-resume-2026-11-28.md) records a
real 256-state repeat and the physical-page alias correction.
Offline `--jobs` and runtime `VICTORY_PIPELINE_WORKERS` are separate controls;
on the four-core RX 570 test machine, use three runtime workers and prepare
previously used objects with `run --execute --precompile --used-only`.
The [fork adaptation report](docs/rx570-fork-ports-2026-11-28.md) lists the
shared frame budgets, fence waits, texture fixes and host-copy reclamation,
with upstream commits, controls and validation limits.
The pipeline recording and source fingerprint follow the approach in
[carlm125/cool_fox](https://github.com/carlm125/cool_fox).

For RX 570 / Polaris experiments:

```powershell
$env:VICTORY_SHADER_PROFILE = "auto"
$env:VICTORY_MESH_FALLBACK = "on"
$env:VICTORY_BVH_PROFILE = "auto"
$env:VICTORY_AMBIENT_FALLBACK = "auto"
$env:VICTORY_MEMORY_PROFILE = "series-s"
$env:VICTORY_TEXTURE_MB = "1536"
$env:VICTORY_WINDOW_TAILS = "1"
$env:VICTORY_PORT_CONSTANTS = "1"
```

Memory profile work is exposed through the console/profile configuration;
see `victory.example.toml` and `docs/architecture.md`.

## Important switches

| Switch | Effect |
|---|---|
| `VICTORY_SHADER_PROFILE=auto/sm65/sm66` | select shader translation profile; auto uses device capability |
| `VICTORY_MESH_FALLBACK=auto/off/on` | select mesh emulation; auto uses it when mesh shaders are unavailable |
| `VICTORY_BVH_PROFILE=auto/full/off/low/static-miss/limited:N` | BVH/RT workload profile; `low` maps to `StaticMiss`, and `auto` selects it on RX 570 |
| `VICTORY_BVH_BUDGET=N` | numeric budget for `limited` |
| `VICTORY_AMBIENT_FALLBACK=off/auto/0.0..1.0` | constant indirect-light fallback; auto enables 0.2 on Polaris |
| `VICTORY_TEXTURE_MB=1536` | sampled-texture budget in MiB, bounded to one quarter of physical VRAM; launcher default for the 8 GiB RX 570 |
| `VICTORY_HOST_COPIES_MB=N` | override the profile's host-copy RAM budget in MiB (default `2560` under `series-s`) |
| `VICTORY_PORT_CONSTANTS=1` | GPU reads and writes a contiguous low-address constants section in place; falls back to copies if opening it fails |
| `VICTORY_WINDOW_TAILS=1` | omit unused prefixes of root-view memory windows |
| `VICTORY_CARD_BUFFERS_MB=768` | launcher budget for buffer copies held in VRAM |
| `VICTORY_D3D12_DRED=1` | report commands in flight on GPU failure |
| `VICTORY_GPU_TIMING=1` | GPU timing diagnostics (use with DRED/diagnostics build) |

## Documentation

- `docs/architecture.md` — translation layer architecture
- `docs/mesh-fallback.md` — compute mesh fallback subset and scheduling
- `docs/fork-ray-tracing.md` — BVH/RT translation status
- `docs/linux-proton.md` — cross-building and running on Linux via Steam Proton
- `docs/asset-policy.md` — no game assets in this repository
- [AMD fork comparison and RX 570 fixes](docs/amd-comparison-2026-11-27.md)
- [RX 570 Shader Model and Agility investigation](docs/rx570-agility.md)
- [RX 570 video-engine timeout investigation](docs/rx570-gpu-timeout-2026-11-27.md)

## AMD work and credits

This fork adapts integer-target blending and pixel quad reads from
[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD),
and shared constants, ordered indirect preparation and submission-time texture
uploads from [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX).
The SM 6.5 barycentrics geometry path is additional Polaris work in this fork.
The comparison document records the source revisions, tests and remaining gaps.

## License

GPL-3.0-or-later (see `LICENSE`). Based on VIctory (Marvisak),
victory-fork (fxceless), and ztrx (Zettrox).
