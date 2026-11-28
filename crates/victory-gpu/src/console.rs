//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.
//!
//! Beside the console model this module owns the **memory profile**: how much
//! title memory the runtime reports and how large the host-side caches may
//! grow. `VICTORY_MEMORY_PROFILE=auto|off|series-s|series-x` (or
//! `[memory].profile` in the config file) selects it; `auto` lowers an 8 GiB
//! card or a 16 GiB PC to the Series S profile. See `victory-core`'s
//! `memory` module for the policy and its unit tests.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

pub use victory_core::memory::{
    DeviceFacts, Limits, MemoryProfile, ProfileRequest, Resolution, Why,
};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure).
    // shortcut: later system updates gave Series S titles more (amount not
    // published here); the title sizes nothing from this figure
    // (it goes by the device type), so update it if a title does.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (0, 7_776 << 20),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

static SERIES_S: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    SERIES_S.store(console == Console::SeriesS, Ordering::Relaxed);
}

pub fn current() -> Console {
    if SERIES_S.load(Ordering::Relaxed) {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}

fn parse_mb(value: &str) -> Option<u64> {
    value
        .parse::<u64>()
        .ok()
        .and_then(|mb| mb.checked_mul(1 << 20))
        .filter(|&bytes| bytes > 0)
}

fn mb_override(name: &str) -> Option<u64> {
    std::env::var(name).ok().as_deref().and_then(parse_mb)
}

fn apply_env_overrides(
    limits: &mut Limits,
    dedicated_video_memory: u64,
    texture_bytes: Option<u64>,
    host_copy_bytes: Option<u64>,
) {
    if let Some(bytes) = texture_bytes {
        let maximum = if dedicated_video_memory > 0 {
            dedicated_video_memory / 4
        } else {
            512 << 20
        };
        limits.texture_bytes = bytes.min(maximum).max(1);
    }
    if let Some(bytes) = host_copy_bytes {
        limits.host_copy_bytes = bytes;
    }
}

/// The profile's caps for this process. Resolved once, from the request and
/// the host's facts, and logged once.
pub fn limits() -> &'static Limits {
    static LIMITS: OnceLock<Limits> = OnceLock::new();
    LIMITS.get_or_init(|| {
        let facts = device_facts();
        let resolved = victory_core::memory::resolve(victory_core::memory::request(), facts);
        let mut limits = Limits::for_profile(resolved.profile, facts);
        apply_env_overrides(
            &mut limits,
            facts.dedicated_video_memory,
            mb_override("VICTORY_TEXTURE_MB"),
            mb_override("VICTORY_HOST_COPIES_MB"),
        );
        tracing::info!(
            profile = limits.profile.as_str(),
            why = resolved.why.as_str(),
            vram_mb = facts.dedicated_video_memory >> 20,
            ram_mb = facts.physical_memory >> 20,
            commit_mb = facts.commit_limit >> 20,
            commit_available_mb = facts.commit_available >> 20,
            title_mb = limits.title_bytes >> 20,
            texture_mb = limits.texture_bytes >> 20,
            texture_single_mb = limits.texture_single_bytes >> 20,
            staging_mb = limits.staging_bytes >> 20,
            tile_pool_mb = limits.tile_pool_bytes >> 20,
            card_buffers_mb = limits.card_buffers_bytes >> 20,
            host_copies_mb = limits.host_copy_bytes >> 20,
            snapshots_mb = limits.snapshot_pool_bytes >> 20,
            descriptor_mb = limits.descriptor_bytes >> 20,
            shader_cache_mb = limits.shader_cache_bytes >> 20,
            rtv_ring = limits.rtv_ring,
            "memory profile"
        );
        limits
    })
}

/// The title memory the runtime reports: the console's split, lowered by the
/// profile. A profile never raises the console's figure.
pub fn effective_title_memory() -> (usize, usize) {
    let (gpu, standard) = current().title_memory();
    let (gpu, standard) = limits().effective_title(gpu as u64, standard as u64);
    (gpu as usize, standard as usize)
}

/// What the host says about its memory. `None` facts (zeros) on other
/// platforms and when a query fails; `auto` then leaves the profile large.
fn device_facts() -> DeviceFacts {
    let mut facts = DeviceFacts {
        dedicated_video_memory: dedicated_video_memory(),
        ..Default::default()
    };
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut status = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: the structure has its own length and is ours to fill.
        if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 {
            facts.physical_memory = status.ullTotalPhys;
            facts.commit_limit = status.ullTotalPageFile;
            facts.commit_available = status.ullAvailPageFile;
        }
    }
    facts
}

/// The largest hardware adapter's dedicated video memory: the PC's own
/// figure for `auto`, not the console's reported split.
#[cfg(windows)]
fn dedicated_video_memory() -> u64 {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory1,
        IDXGIFactory4,
    };
    use windows::core::Interface;
    // SAFETY: plain DXGI factory and adapter queries; any refusal is zero.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return 0;
        };
        let Ok(factory) = factory.cast::<IDXGIFactory1>() else {
            return 0;
        };
        let mut most = 0u64;
        for index in 0..8 {
            let Ok(adapter) = factory.EnumAdapters1(index) else {
                break;
            };
            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            most = most.max(desc.DedicatedVideoMemory as u64);
        }
        most
    }
}

#[cfg(not(windows))]
fn dedicated_video_memory() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reported_title_never_exceeds_the_profile() {
        let limits = Limits::for_profile(MemoryProfile::SeriesS, DeviceFacts::default());
        let (gpu, standard) = Console::SeriesX.title_memory();
        let (gpu, standard) = limits.effective_title(gpu as u64, standard as u64);
        assert_eq!((gpu, standard), (0, victory_core::memory::SERIES_S_TITLE));
        assert!(gpu + standard <= limits.title_bytes);
    }

    #[test]
    fn the_console_figure_survives_without_a_profile() {
        let limits = Limits::for_profile(MemoryProfile::Off, DeviceFacts::default());
        for console in [Console::SeriesX, Console::SeriesS] {
            let (gpu, standard) = console.title_memory();
            assert_eq!(
                limits.effective_title(gpu as u64, standard as u64),
                (gpu as u64, standard as u64)
            );
        }
    }

    #[test]
    fn auto_resolution_is_the_policy_of_record() {
        let facts = DeviceFacts {
            dedicated_video_memory: 8 * (1 << 30),
            ..Default::default()
        };
        let resolved = victory_core::memory::resolve(ProfileRequest::Auto, facts);
        assert_eq!(resolved.profile, MemoryProfile::SeriesS);
        assert_eq!(resolved.why, Why::DedicatedVideoMemory);
    }

    #[test]
    fn host_copies_override_lifts_the_series_s_cap_only_when_positive() {
        let mut limits = Limits::for_profile(
            MemoryProfile::SeriesS,
            DeviceFacts {
                dedicated_video_memory: 8 * (1 << 30),
                ..Default::default()
            },
        );
        let original = limits.host_copy_bytes;
        assert_eq!(original, 5 * (1 << 30) / 2);
        assert_eq!(parse_mb("0"), None);
        assert_eq!(parse_mb("bad"), None);
        assert_eq!(parse_mb("4096"), Some(4096 << 20));

        apply_env_overrides(&mut limits, 8 * (1 << 30), None, parse_mb("0"));
        assert_eq!(limits.host_copy_bytes, original);

        apply_env_overrides(&mut limits, 8 * (1 << 30), None, parse_mb("4096"));
        assert_eq!(limits.host_copy_bytes, 4096 << 20);
    }
}
