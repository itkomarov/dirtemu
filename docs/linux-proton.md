# Running on Linux with Proton

`dirtemu` builds for Windows on Linux with `cargo-xwin` and runs under
**Steam Proton** (`vkd3d-proton` for Direct3D 12 and `DXVK` for `DXGI`). Use
Proton rather than bare system Wine: Proton bundles `vkd3d-proton 3.1+` and
recent `vkd3d-shader`, which avoids SPIR-V pipeline crashes in Mesa RADV and
compiles the helper compute/graphics shaders without a Windows `d3dcompiler_47.dll`.

## Requirements

- Steam **Proton Experimental** / **Proton 10.0+** (or **GE-Proton11+**,
  providing `vkd3d-proton 3.1+`). Older Proton builds can emit invalid SPIR-V
  and crash RADV during `vkCreateGraphicsPipelines`.
- A Vulkan 1.3/1.4 driver:
  - **RDNA 2+** (`RADV`): hardware mesh shaders (`VICTORY_MESH_FALLBACK=auto`)
    and SM 6.6.
  - **RX 570 / Polaris** (`RADV`): runs through the compute mesh fallback
    (`VICTORY_MESH_FALLBACK=on`). Mesa RADV on Polaris exposes
    `VK_EXT_shader_image_atomic_int64`, `VK_EXT_mutable_descriptor_type`, and
    `VK_EXT_descriptor_heap`, so Linux users can run `VICTORY_SHADER_PROFILE=sm66`
    (with `VKD3D_SHADER_MODEL=6_6` if needed by their `vkd3d-proton` build),
    matching the Agility SM 6.6 path documented in [`rx570-agility.md`](rx570-agility.md).
- `rustup`, `lld`, and [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin):

```sh
rustup toolchain install 1.98.1 --profile minimal -c clippy -c rustfmt -t x86_64-pc-windows-msvc
cargo install --locked cargo-xwin
```

## Build

```sh
export XWIN_ACCEPT_LICENSE=1
cargo +1.98.1 xwin build --release -p victory --target x86_64-pc-windows-msvc
```

Diagnostics build:

```sh
cargo +1.98.1 xwin build --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc
```

## Run

[`scripts/proton-run.sh`](../scripts/proton-run.sh) finds Steam Proton
automatically (or uses `PROTON=...`), initializes a prefix under
`~/.local/share/dirtemu/proton` (override with `PREFIX=...`), and sets
`VKD3D_SHADER_CACHE_PATH` inside that prefix:

```sh
scripts/proton-run.sh config
scripts/proton-run.sh precompile /path/to/game/<the game's exe> --console series-x --jobs 3
VICTORY_MEMORY_PROFILE=series-s \
VICTORY_BVH_PROFILE=auto \
VICTORY_AMBIENT_FALLBACK=auto \
scripts/proton-run.sh run /path/to/game/<the game's exe> --console series-x --execute
```

For **4 GiB RX 570** cards on Linux, keep `VICTORY_TEXTURE_MB` low (for
example `768` or `1024`; `console.rs` clamps it to at most one quarter of
physical VRAM). On a **16 GiB RAM** Linux host, make sure swap is enabled for
large world loads, or tune `VICTORY_HOST_COPIES_MB` if host buffer copies hit
the default `2,560 MiB` `series-s` cap.

## Tests

Run Windows unit/GPU tests under Proton:

```sh
CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER="$PWD/scripts/proton-run.sh" \
    cargo +1.98.1 xwin test --workspace --target x86_64-pc-windows-msvc
```

`xtask`'s `repository_passes_policy` spawns Linux `git`; run that check natively:

```sh
cargo +1.98.1 test -p xtask --locked
```

## Proton differences handled in tree

- `host.rs`, `compose.rs`, `indirect.rs`: helper HLSL shaders omit
  `[RootSignature]` attributes and `RWByteAddressBuffer` methods (unsupported in
  Proton's `vkd3d-shader`) in favor of `StructuredBuffer<uint>` /
  `RWStructuredBuffer<uint>` and programmatic root signatures.
- `pipeline/library.rs`: empty `CreatePipelineLibrary` passes a null pointer
  instead of a dangling empty-slice pointer (`E_INVALIDARG` in `vkd3d-proton`).
- `victory-shader/src/emit/lower.rs`: a single shared returning block per
  function avoids unstructured SPIR-V crashes in `dxil-spirv` / RADV.
- `victory-hle/src/modules/net.rs`: `WSAConnectByList` and `GetAddrInfoExA`
  return unsupported / not-found when absent from Proton's `ws2_32.dll`.
- `watch.rs`: under Wine/Proton, `VirtualQuery` / `GetWriteWatch` results for
  title allocations and `MEM_MAPPED` views are cached per submission to reduce
  VM lock contention.
