#!/usr/bin/env bash
# Runs victory.exe (or a cargo test binary) under Steam Proton on Linux,
# using vkd3d-proton (D3D12 -> Vulkan) and DXVK (DXGI).
# See docs/linux-proton.md.
#
#   scripts/proton-run.sh config
#   scripts/proton-run.sh precompile /path/to/game/<exe> --console series-s --jobs 3
#   scripts/proton-run.sh run /path/to/game/<exe> --console series-x --execute
#
# As cargo's Windows test runner:
#   CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=$PWD/scripts/proton-run.sh \
#       cargo xwin test --workspace --target x86_64-pc-windows-msvc
#
# Environment:
#   PROTON       Proton directory (default: Steam Proton Experimental/11.0/10.0 or GE/UMU-Proton in compatibilitytools.d)
#   PREFIX       Proton prefix directory (default: ~/.local/share/dirtemu/proton)
#   VICTORY_EXE  exe to launch when the first argument is a victory subcommand
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

has_proton() {
    local dir="$1"
    [[ -x "$dir/files/bin/wine" && -f "$dir/files/lib/wine/vkd3d-proton/x86_64-windows/d3d12core.dll" ]]
}

find_proton() {
    local base name dir
    for base in \
        "$HOME/.steam/steam/steamapps/common" \
        "$HOME/.local/share/Steam/steamapps/common" \
        "$HOME/.var/app/com.valvesoftware.Steam/data/Steam/steamapps/common"; do
        for name in "Proton - Experimental" "Proton 11.0" "Proton 10.0"; do
            if has_proton "$base/$name"; then
                printf '%s\n' "$base/$name"
                return 0
            fi
        done
    done
    for base in \
        "$HOME/.steam/root/compatibilitytools.d" \
        "$HOME/.steam/steam/compatibilitytools.d" \
        "$HOME/.local/share/Steam/compatibilitytools.d" \
        "$HOME/.var/app/com.valvesoftware.Steam/data/Steam/compatibilitytools.d" \
        "/usr/share/steam/compatibilitytools.d"; do
        [[ -d "$base" ]] || continue
        while IFS= read -r dir; do
            if has_proton "$dir"; then
                printf '%s\n' "$dir"
                return 0
            fi
        done < <(find "$base" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) -iname '*proton*' | sort -Vr)
    done
    return 1
}

if [[ -z "${PROTON:-}" ]]; then
    if ! PROTON="$(find_proton)"; then
        echo "Steam Proton (Experimental / 10.0+ / GE-Proton) not found; install it or set PROTON=/path/to/Proton" >&2
        exit 1
    fi
fi
if ! has_proton "$PROTON"; then
    echo "Invalid PROTON directory: $PROTON (missing files/bin/wine or vkd3d-proton/d3d12core.dll)" >&2
    exit 1
fi

PREFIX="${PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/dirtemu/proton}"

if [[ ! -f "$PREFIX/pfx/drive_c/windows/system32/d3d12core.dll" || ! -f "$PREFIX/pfx/system.reg" ]]; then
    mkdir -p "$PREFIX"
    steam_root="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-}"
    if [[ -z "$steam_root" ]]; then
        parent="$PROTON"
        while [[ "$parent" != "/" ]]; do
            parent="$(dirname "$parent")"
            if [[ -d "$parent/steamapps" || -d "$parent/compatibilitytools.d" ]]; then
                steam_root="$parent"
                break
            fi
        done
        steam_root="${steam_root:-$HOME/.local/share/Steam}"
    fi
    STEAM_COMPAT_DATA_PATH="$PREFIX" \
    STEAM_COMPAT_CLIENT_INSTALL_PATH="$steam_root" \
    SteamAppId=0 SteamGameId=0 \
        python3 "$PROTON/proton" run cmd /c exit >/dev/null
fi

export WINEPREFIX="$PREFIX/pfx"
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3d11,d3d10core=n,b;${WINEDLLOVERRIDES:-}"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEFSYNC=1
export VKD3D_SHADER_CACHE_PATH="${VKD3D_SHADER_CACHE_PATH:-$PREFIX/vkd3d-cache}"
mkdir -p "$VKD3D_SHADER_CACHE_PATH"

WINE="$PROTON/files/bin/wine"
if [[ -n "${VICTORY_EXE:-}" ]]; then
    exec "$WINE" "$VICTORY_EXE" "$@"
elif [[ "${1:-}" == *.exe && -f "${1:-}" ]]; then
    exec "$WINE" "$@"
else
    exec "$WINE" "$ROOT/target/x86_64-pc-windows-msvc/release/victory.exe" "$@"
fi
