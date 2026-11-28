//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::{RtvRecord, View};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::{HostPipeline, Stage};

mod ambient;
mod mesh;

#[derive(Clone, Copy)]
struct BufferWarmLimits {
    span: std::time::Duration,
    bytes: u64,
}

impl BufferWarmLimits {
    fn configured() -> &'static Self {
        static LIMITS: std::sync::OnceLock<BufferWarmLimits> = std::sync::OnceLock::new();
        LIMITS.get_or_init(|| {
            let var = |name: &str| std::env::var(name).ok()?.trim().parse::<u64>().ok();
            let span = var("VICTORY_BUFFER_WARM_US")
                .map(std::time::Duration::from_micros)
                .unwrap_or_else(|| {
                    std::time::Duration::from_millis(var("VICTORY_BUFFER_WARM_MS").unwrap_or(16))
                });
            let bytes = var("VICTORY_BUFFER_WARM_BYTES").unwrap_or_else(|| {
                var("VICTORY_BUFFER_WARM_MB")
                    .unwrap_or(32)
                    .saturating_mul(1 << 20)
            });
            tracing::info!(
                micros = span.as_micros() as u64,
                bytes,
                "bindless buffer warmup budget per presented frame"
            );
            BufferWarmLimits { span, bytes }
        })
    }
}

#[derive(Default)]
struct BufferWarmBudget {
    frame: Option<u64>,
    elapsed: std::time::Duration,
    bytes: u64,
    busy: bool,
    last_served: u64,
    priority: Option<u64>,
    priority_done: bool,
    active: std::collections::BTreeMap<u64, (std::sync::Weak<()>, u64)>,
    known: std::collections::HashMap<u64, std::sync::Weak<crate::shadow::Shadow>>,
    ready: BufferReadyCoverage,
}

#[derive(Default)]
struct BufferReadyCoverage {
    presented: Option<u64>,
    copies: std::collections::HashMap<(u64, u64), Vec<(u64, u64)>>,
    full: std::collections::HashMap<u64, (std::sync::Weak<crate::shadow::Shadow>, u64)>,
}

impl BufferReadyCoverage {
    fn advance(&mut self, presented: u64) -> bool {
        if self.presented.is_some_and(|current| current > presented) {
            return false;
        }
        if self.presented != Some(presented) {
            self.presented = Some(presented);
            self.copies.clear();
            self.full.retain(|_, (copy, _)| copy.strong_count() != 0);
        }
        true
    }

    fn allow_full(
        &mut self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        presented: u64,
        interval: u64,
    ) -> bool {
        if !self.advance(presented) {
            return true;
        }
        if let Some((_, last)) = self.full.get(&shadow.id())
            && *last != presented
            && presented < last.saturating_add(interval)
        {
            return false;
        }
        self.full
            .insert(shadow.id(), (std::sync::Arc::downgrade(shadow), presented));
        true
    }

    fn claim(
        &mut self,
        recording: u64,
        copy: u64,
        presented: u64,
        from: u64,
        to: u64,
    ) -> Vec<(u64, u64)> {
        if from >= to {
            return Vec::new();
        }
        if !self.advance(presented) {
            return vec![(from, to)];
        }
        let covered = self.copies.entry((recording, copy)).or_default();
        let mut missing = Vec::new();
        let mut at = from;
        for &(a, b) in covered.iter() {
            if b <= at {
                continue;
            }
            if a >= to {
                break;
            }
            if a > at {
                missing.push((at, a.min(to)));
            }
            at = at.max(b);
            if at >= to {
                break;
            }
        }
        if at < to {
            missing.push((at, to));
        }
        let first = covered.partition_point(|&(_, end)| end < from);
        let mut end = first;
        let mut merged = (from, to);
        while let Some(&(a, b)) = covered.get(end) {
            if a > merged.1 {
                break;
            }
            merged.0 = merged.0.min(a);
            merged.1 = merged.1.max(b);
            end += 1;
        }
        covered.splice(first..end, [merged]);
        missing
    }
}

impl BufferWarmBudget {
    fn shared() -> &'static std::sync::Mutex<Self> {
        static BUDGET: std::sync::OnceLock<std::sync::Mutex<BufferWarmBudget>> =
            std::sync::OnceLock::new();
        BUDGET.get_or_init(Default::default)
    }

    fn begin<'a>(
        budget: &'a std::sync::Mutex<Self>,
        heap: u64,
        life: &std::sync::Arc<()>,
        frame: u64,
        limits: BufferWarmLimits,
    ) -> Option<BufferWarmPermit<'a>> {
        use std::ops::Bound::{Excluded, Unbounded};
        let mut held = budget.lock().unwrap_or_else(|e| e.into_inner());
        if held.frame.is_some_and(|current| current > frame) {
            return None;
        }
        held.active
            .retain(|_, (life, seen)| life.strong_count() != 0 && seen.saturating_add(1) >= frame);
        held.active
            .insert(heap, (std::sync::Arc::downgrade(life), frame));
        if held.busy {
            return None;
        }
        if held.frame != Some(frame) {
            held.frame = Some(frame);
            held.elapsed = std::time::Duration::ZERO;
            held.bytes = 0;
            held.priority_done = false;
            held.priority = held
                .active
                .range((Excluded(held.last_served), Unbounded))
                .next()
                .or_else(|| held.active.first_key_value())
                .map(|(&id, _)| id);
            held.known.retain(|_, copy| copy.strong_count() != 0);
        }
        if held
            .priority
            .is_some_and(|id| !held.active.contains_key(&id))
        {
            held.priority = Some(heap);
        }
        if limits.span.is_zero()
            || limits.bytes == 0
            || held.elapsed >= limits.span
            || held.bytes >= limits.bytes
            || (!held.priority_done && held.priority != Some(heap))
        {
            return None;
        }
        held.busy = true;
        held.priority_done = true;
        drop(held);
        Some(BufferWarmPermit {
            budget,
            heap,
            frame,
            began: std::time::Instant::now(),
            copy: None,
        })
    }

    fn forget(&mut self, heap: u64) {
        self.active.remove(&heap);
    }
}

struct BufferWarmPermit<'a> {
    budget: &'a std::sync::Mutex<BufferWarmBudget>,
    heap: u64,
    frame: u64,
    began: std::time::Instant,
    copy: Option<(u64, u64, std::sync::Weak<crate::shadow::Shadow>)>,
}

impl BufferWarmPermit<'_> {
    fn copied(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) -> u64 {
        let bytes = unsafe { shadow.buffer.GetDesc() }.Width;
        let fresh = !self
            .budget
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .known
            .contains_key(&shadow.id());
        self.copy = Some((shadow.id(), bytes, std::sync::Arc::downgrade(shadow)));
        if fresh { bytes } else { 0 }
    }
}

impl Drop for BufferWarmPermit<'_> {
    fn drop(&mut self) {
        let mut held = self.budget.lock().unwrap_or_else(|e| e.into_inner());
        if held.frame == Some(self.frame) {
            held.elapsed = held.elapsed.saturating_add(self.began.elapsed());
            if let Some((id, bytes, copy)) = self.copy.take()
                && held.known.insert(id, copy).is_none()
            {
                held.bytes = held.bytes.saturating_add(bytes);
            }
        }
        held.last_served = self.heap;
        held.busy = false;
    }
}

fn buffer_record(heap: &crate::descriptor::Heap, index: u32) -> Option<[u32; 3]> {
    let handle = heap.handle(index)?;
    Some(unsafe { (handle as *const [u32; 3]).read_unaligned() })
}

struct BufferReapPass<'a>(&'a crate::descriptor::host::HostHeap);

impl Drop for BufferReapPass<'_> {
    fn drop(&mut self) {
        self.0.buffer_ranges().reaping = false;
    }
}

fn reap_buffer_ranges(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap) {
    let epoch = crate::descriptor::release_epoch();
    let started = std::time::Instant::now();
    {
        let mut cache = host.buffer_ranges();
        if cache.last_reaped_epoch >= epoch || cache.reaping {
            return;
        }
        cache.reaping = true;
    }
    let _reaping = BufferReapPass(host);
    let (records, revision) = {
        let cache = host.buffer_ranges();
        let records = cache
            .ranges
            .iter()
            .filter(|(_, range)| crate::descriptor::is_object_record(&range.3))
            .map(|(&index, range)| (index, range.3))
            .collect::<Vec<_>>();
        (records, cache.object_revision)
    };
    let expired = crate::descriptor::expired_object_views(&records);
    let before = crate::shadow::live_copy_bytes();
    let mut invalidated = 0u64;
    let mut gone = Vec::new();
    for (index, expected) in expired {
        if let Some(handle) = heap.handle(index)
            && host.invalidate_released(index, expected, handle)
        {
            invalidated += 1;
        }
        let mut cache = host.buffer_ranges();
        if cache
            .ranges
            .get(&index)
            .is_some_and(|range| u64::from(range.3[2]) << 32 | u64::from(range.3[1]) == expected)
            && let Some(range) = cache.ranges.remove(&index)
        {
            gone.push(range.0);
        }
    }
    let stale_slots = gone.len() as u64;
    let mut affected = std::collections::HashMap::new();
    for shadow in &gone {
        affected
            .entry(shadow.id())
            .or_insert_with(|| unsafe { shadow.buffer.GetDesc() }.Width);
    }
    let affected_bytes: u64 = affected.values().sum();
    drop(gone);
    let after = crate::shadow::live_copy_bytes();
    let delta = if stale_slots != 0 || invalidated != 0 {
        (i128::from(before) - i128::from(after)).clamp(i128::from(i64::MIN), i128::from(i64::MAX))
            as i64
    } else {
        0
    };
    let mut cache = host.buffer_ranges();
    if cache.object_revision == revision {
        cache.last_reaped_epoch = cache.last_reaped_epoch.max(epoch);
    }
    cache.reap_passes += 1;
    cache.stale_slots += stale_slots;
    cache.stale_host_slots += invalidated;
    cache.stale_shadow_bytes = cache.stale_shadow_bytes.saturating_add(affected_bytes);
    cache.reap_us = cache
        .reap_us
        .saturating_add(started.elapsed().as_micros() as u64);
    cache.reap_sampled_live_change = delta;
    cache.reap_live_before = before;
    cache.reap_live_after = after;
}

fn log_buffer_warmup(host: &crate::descriptor::host::HostHeap) {
    let Ok(mut cache) = host.buffer_ranges_try() else {
        return;
    };
    let now = crate::host::now_ms();
    if cache.logged_ms != 0 && now.saturating_sub(cache.logged_ms) < 5000 {
        return;
    }
    cache.logged_ms = now.max(1);
    tracing::info!(heap = host.id(), total = host.buffer_slot_count(), ready = cache.ranges.len(), pending = cache.work.pending(), retry = host.buffer_retry_due(u64::MAX).len(), scanning = cache.work.has_work(), cursor = ?cache.work.cursor(), scanned = cache.scanned, admitted_copy_mb = cache.admitted_bytes >> 20, warm_us = cache.warm_us, ready_refresh_us = cache.refresh_us, full_ready_batches = cache.full_ready_batches, cold_jobs = cache.cold_jobs, ready_bytes = cache.ready_bytes, ready_ranges = cache.ready_ranges, stale_slots = cache.stale_slots, stale_host_slots = cache.stale_host_slots, stale_shadow_bytes = cache.stale_shadow_bytes, reap_us = cache.reap_us, reap_passes = cache.reap_passes, reap_sampled_live_change = cache.reap_sampled_live_change, reap_live_before = cache.reap_live_before, reap_live_after = cache.reap_live_after, presented = crate::queue::presented(), "bindless buffer warmup progress");
}

pub(crate) const BAKE_PIXEL_SHADERS: [&str; 4] = [
    "c0a7fe77b1862e10",
    "0d1575a396dfe513",
    "b7cfd8dda91ebb6b",
    "280967ad158cee0b",
];

struct TextureStageLimits {
    shared: bool,
    span: std::time::Duration,
    bytes: u64,
}

impl TextureStageLimits {
    fn configured() -> &'static Self {
        static LIMITS: std::sync::OnceLock<TextureStageLimits> = std::sync::OnceLock::new();
        LIMITS.get_or_init(|| {
            let var = |name: &str| std::env::var(name).ok()?.trim().parse::<u64>().ok();
            let shared =
                std::env::var_os("VICTORY_TEXTURE_STAGE_PER_FRAME").is_none_or(|v| v != "0");
            let span = var("VICTORY_TEXTURE_STAGE_US")
                .map(std::time::Duration::from_micros)
                .unwrap_or_else(|| {
                    std::time::Duration::from_millis(
                        var("VICTORY_TEXTURE_STAGE_MS").unwrap_or(if shared { 6 } else { 250 }),
                    )
                });
            let bytes = var("VICTORY_TEXTURE_STAGE_BYTES").unwrap_or_else(|| {
                var("VICTORY_TEXTURE_STAGE_MB")
                    .unwrap_or(192)
                    .saturating_mul(1 << 20)
            });
            tracing::info!(
                shared,
                micros = span.as_micros() as u64,
                bytes,
                "texture staging budget"
            );
            TextureStageLimits {
                shared,
                span,
                bytes,
            }
        })
    }
}

struct TextureStageBudget {
    frame: u64,
    elapsed: std::time::Duration,
}

impl TextureStageBudget {
    const fn new() -> Self {
        Self {
            frame: u64::MAX,
            elapsed: std::time::Duration::ZERO,
        }
    }

    fn begin(
        budget: &std::sync::Mutex<Self>,
        frame: u64,
        shared: bool,
        span: std::time::Duration,
    ) -> Option<TextureStagePass<'_>> {
        let mut held = if shared {
            let mut held = match budget.try_lock() {
                Ok(held) => held,
                Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => return None,
            };
            if held.frame != u64::MAX && held.frame > frame {
                return None;
            }
            if held.frame != frame {
                held.frame = frame;
                held.elapsed = std::time::Duration::ZERO;
            }
            Some(held)
        } else {
            None
        };
        let left = span.saturating_sub(
            held.as_mut()
                .map_or(std::time::Duration::ZERO, |held| held.elapsed),
        );
        if left.is_zero() {
            return None;
        }
        Some(TextureStagePass {
            held,
            began: std::time::Instant::now(),
            left,
        })
    }
}

struct TextureStagePass<'a> {
    held: Option<std::sync::MutexGuard<'a, TextureStageBudget>>,
    began: std::time::Instant,
    left: std::time::Duration,
}

impl TextureStagePass<'_> {
    fn exhausted(&self) -> bool {
        self.began.elapsed() >= self.left
    }
}

impl Drop for TextureStagePass<'_> {
    fn drop(&mut self) {
        if let Some(held) = &mut self.held {
            held.elapsed = held.elapsed.saturating_add(self.began.elapsed());
        }
    }
}

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours, until one of these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_none_or(|v| v != "0"))
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth views, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_views: Vec<RtvRecord>,
    depth_view: Option<RtvRecord>,
    targets_set: bool,
    host_targets: Option<bool>,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    mesh_group_base: u32,
    mesh_workspaces: std::collections::HashMap<usize, mesh::Workspace>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u64, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
    tables_expired: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, id, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, id, first, n);
        }
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_expired(&self) -> bool {
        self.tables_expired
    }

    pub(crate) fn tables_done(&mut self) {
        self.tables_expired |= !self.dynamic.is_empty();
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!("a list ran again after its descriptor tables were given back")
            });
        }
        for (heap, id, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, id, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((
            parts.next()?.to_owned(),
            parts.next().unwrap_or("").to_owned(),
        ))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15])
        && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow.left(address).min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = host_topology(topology);
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self
            .vertex
            .iter_mut()
            .skip(start as usize)
            .take(count as usize)
        {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    pub(crate) fn set_targets(&mut self, _list: &HostList, handles: &[u64], depth: u64) {
        self.target_views.clear();
        for &handle in handles {
            match crate::descriptor::rtv_record(handle) {
                Some(record) => self.target_views.push(record),
                None => once!(
                    handle = format_args!("{handle:#x}"),
                    "render target without a recorded view"
                ),
            }
        }
        self.depth_view = crate::descriptor::dsv_record(depth);
        self.targets_set = true;
        self.invalidate_targets();
    }

    fn invalidate_targets(&mut self) {
        self.host_targets = None;
        self.targets.clear();
        self.depth = 0;
    }

    fn bind_targets(&mut self, list: &HostList, depth_stencil: bool) {
        if !self.targets_set || self.host_targets == Some(depth_stencil) {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for record in &self.target_views {
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = depth_stencil
            .then_some(self.depth_view)
            .flatten()
            .and_then(|record| {
                let texture = crate::resource::host_resource(record.resource)?;
                self.depth = record.resource as usize;
                self.depth_state = depth_state(texture, record.depth_desc().as_ref());
                if crate::resource::depth_and_unordered(record.resource) {
                    crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
                }
                #[cfg(feature = "diagnostics")]
                crate::capture::note_target(record.resource);
                Some(host.dsv(texture, record.depth_desc().as_ref()))
            });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
        self.host_targets = Some(depth_stencil);
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        self.invalidate_targets();
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{
            USER_MEMORY_BASE, USER_MESH_GROUP_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS,
        };
        if kind == USER_MESH_GROUP_BASE {
            return self.mesh_group_base;
        }
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(
                slot,
                dword,
                "a shader takes the address of a buffer view that is not there: it reads 0"
            );
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(
        &self,
        list: &HostList,
        compute: bool,
        writes: bool,
    ) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Binds a stage's SM 6.5 descriptor aliases: each root parameter is a
    /// table over its whole heap, so the list binds the heap's base (the
    /// shader's register number is the heap index). False if a heap that an
    /// alias reads is not bound: the draw must not run with an undefined
    /// table.
    fn alias_tables(&self, list: &HostList, stage: &Stage, compute: bool) -> bool {
        for &(param, sampler) in &stage.aliases {
            let title = self.heaps[usize::from(sampler)];
            let host = (title != 0)
                .then(|| crate::descriptor::heap(title as This).host())
                .flatten();
            let Some(host) = host else {
                once!(
                    param,
                    sampler,
                    "a shader reads a descriptor alias whose heap the title did not bind: the draw is dropped"
                );
                return false;
            };
            let handle = host.gpu_handle(0);
            // SAFETY: an open list; the table is over the heap the title
            // bound, and the list keeps that heap alive.
            unsafe {
                if compute {
                    list.list.SetComputeRootDescriptorTable(param, handle);
                } else {
                    list.list.SetGraphicsRootDescriptorTable(param, handle);
                }
            }
        }
        true
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        self.bind_targets(list, p.depth_stencil);
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let title_compute = mesh::title_compute(compute, p.mesh_fallback.is_some());
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && p.depth_stencil && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = if compute {
            p.mesh_fallback
                .as_ref()
                .map_or(&p.root, |fallback| &fallback.compute_root)
        } else {
            &p.root
        };
        let root_id = root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root_id {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(root);
                } else {
                    l.SetGraphicsRootSignature(root);
                }
            }
            self.host_roots[usize::from(compute)] = root_id;
        }
        let pso = if compute {
            p.mesh_fallback
                .as_ref()
                .map_or(&p.pso, |fallback| &fallback.compute)
        } else {
            &p.pso
        };
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(pso) };
            self.host_pipeline = pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(pso);
            list.keep(root);
        }
        for stage in &p.stages {
            if p.mesh_fallback.is_some()
                && ((compute && stage.shader.stage != "mesh")
                    || (!compute && stage.shader.stage == "mesh"))
            {
                continue;
            }
            let t = stage.translated();
            // SM 6.5: each descriptor alias is a table over the whole
            // bound heap, starting at its base (the shader's register is
            // the heap index). The heaps and the root signature are set.
            if !self.alias_tables(list, stage, compute) {
                return false;
            }
            STORES.set(t.stores);
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, title_compute);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                return false;
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t
                .host
                .root_srvs
                .contains(&victory_shader::emit::SLOT_MEMORY)
            {
                let Some(ready) = self.memory_ready(list, title_compute, writes) else {
                    once!(
                        "a shader that reads title memory by address without a view heap to hold its views: the draw is dropped"
                    );
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(title_compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(title_compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => {
                                once!("a shader picks buffers from a view heap that is not set")
                            }
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) =
                        self.roots_ref(title_compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        once!(
                            slot,
                            address = format_args!("{address:#x}"),
                            "a required root buffer could not be prepared: draw or dispatch dropped"
                        );
                        return false;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            title_compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(title_compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(
                            slot,
                            "a shader writes through a root view the title did not set"
                        );
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                if address.is_some() && shadow.is_none() {
                    once!("a required root UAV buffer is unavailable: draw or dispatch dropped");
                    return false;
                }
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, host.id(), index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(
                        index,
                        &s.buffer,
                        offset - bias,
                        s.left(*a).min(limit) + bias,
                    );
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                if address.is_some() && shadow.is_none() {
                    once!("a required root SRV buffer is unavailable: draw or dispatch dropped");
                    return false;
                }
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, host.id(), index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        self.heap_buffers_at_frame(list, crate::frame::current())
    }

    fn heap_buffers_at_frame(&self, list: &HostList, frame: u64) -> Option<u64> {
        self.heap_buffers_budgeted(
            list,
            frame,
            crate::queue::presented(),
            BufferWarmBudget::shared(),
            *BufferWarmLimits::configured(),
        )
    }

    fn heap_buffers_budgeted(
        &self,
        list: &HostList,
        frame: u64,
        presented: u64,
        budget: &std::sync::Mutex<BufferWarmBudget>,
        limits: BufferWarmLimits,
    ) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        reap_buffer_ranges(heap, host);
        if !list.first_heap_use(host.id()) {
            log_buffer_warmup(host);
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        let started = std::time::Instant::now();
        // Per host heap: the host copy and range each slot's buffer view
        // reads, kept from pass to pass. Only the slots written since the
        // last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
        // at best); all of them every 16 passes, because a view of a
        // released resource goes stale without a write, and a slot written
        // before its memory had a host copy is only made then.
        let mut changed = host.buffer_changes();
        changed.extend(
            host.buffer_retry_due(presented)
                .into_iter()
                .filter(|&index| {
                    heap.handle(index)
                        .and_then(crate::descriptor::view)
                        .as_ref()
                        .and_then(crate::descriptor::host::buffer_range)
                        .is_some()
                }),
        );
        let (fresh_ready, life) = {
            let mut cache = host.buffer_ranges();
            let fresh = cache.last_frame != Some(frame);
            if fresh {
                cache.last_frame = Some(frame);
                if cache.passes.is_multiple_of(16) {
                    cache.work.scan();
                }
                cache.passes += 1;
            }
            let fresh_ready = cache
                .last_ready_presented
                .is_none_or(|last| presented > last);
            if fresh_ready {
                cache.last_ready_presented = Some(presented);
                cache.full_ready_batches += 1;
            }
            for index in changed {
                if cache
                    .ranges
                    .get(&index)
                    .is_some_and(|range| buffer_record(heap, index) != Some(range.3))
                {
                    cache.ranges.remove(&index);
                }
                cache.work.enqueue(index);
            }
            (fresh_ready, cache.life.clone())
        };
        let warming = std::time::Instant::now();
        let mut scanned = 0u64;
        let mut admitted_bytes = 0u64;
        let mut cold_jobs = 0u64;
        let mut newly_ready = Vec::new();
        let mut new_slots = std::collections::HashSet::new();
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive =
            !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        loop {
            if !host.buffer_ranges().work.has_work() {
                budget
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .forget(host.id());
                break;
            }
            let Some(mut permit) =
                BufferWarmBudget::begin(budget, host.id(), &life, presented, limits)
            else {
                break;
            };
            let index = host
                .buffer_ranges()
                .work
                .next(|cursor| host.buffer_slot_after(cursor));
            let Some(index) = index else {
                budget
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .forget(host.id());
                break;
            };
            scanned += 1;
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            let Some(record) = buffer_record(heap, index) else {
                continue;
            };
            let current = crate::descriptor::view(handle);
            if current.is_none() {
                host.mirror_at_present(index, handle, true, presented);
                host.buffer_ranges().ranges.remove(&index);
                continue;
            }
            if current
                .as_ref()
                .and_then(crate::descriptor::host::buffer_range)
                .is_none()
            {
                host.buffer_ranges().ranges.remove(&index);
                continue;
            }
            if host
                .buffer_ranges()
                .ranges
                .get(&index)
                .is_some_and(|range| range.3 == record)
                && host.buffer_slot_ready(index)
            {
                continue;
            }
            cold_jobs += 1;
            let Some(view) = host.mirror_at_present(index, handle, true, presented) else {
                continue;
            };
            if host.buffer_retry_pending(index) || !host.buffer_slot_ready(index) {
                continue;
            }
            // Every 64 views while the device lives (a cheap query).
            if alive
                && scanned.is_multiple_of(64)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                tracing::error!(
                    why,
                    slot = index,
                    "the host device went while buffer views of the heap were made"
                );
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
                continue;
            };
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            admitted_bytes = admitted_bytes.saturating_add(permit.copied(&shadow));
            let mut cache = host.buffer_ranges();
            if buffer_record(heap, index) != Some(record) {
                cache.ranges.remove(&index);
                cache.work.enqueue(index);
                continue;
            }
            cache
                .ranges
                .insert(index, (shadow.clone(), address, address + len, record));
            if crate::descriptor::is_object_record(&record) {
                cache.object_revision = cache.object_revision.wrapping_add(1);
                cache.last_reaped_epoch = 0;
            }
            new_slots.insert(index);
            newly_ready.push((shadow, address, address + len));
        }
        let refreshing = std::time::Instant::now();
        let mut kept = host.buffer_ranges();
        kept.scanned += scanned;
        kept.cold_jobs += cold_jobs;
        kept.warm_us = kept
            .warm_us
            .saturating_add(warming.elapsed().as_micros() as u64);
        kept.admitted_bytes = kept.admitted_bytes.saturating_add(admitted_bytes);
        let views = kept.ranges.len() as u32;
        // Per host copy, the span its picked views cover.
        let mut spans: std::collections::HashMap<
            usize,
            (std::sync::Arc<crate::shadow::Shadow>, u64, u64),
        > = std::collections::HashMap::new();
        let mut note = |shadow: &std::sync::Arc<crate::shadow::Shadow>, from: u64, to: u64| {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            let span = spans
                .entry(key)
                .or_insert_with(|| (shadow.clone(), from, to));
            span.1 = span.1.min(from);
            span.2 = span.2.max(to);
        };
        if fresh_ready {
            for (index, (shadow, from, to, _)) in &kept.ranges {
                if !new_slots.contains(index) {
                    note(shadow, *from, *to);
                }
            }
        }
        drop(kept);
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every 120 frames (unmapped pages read
        // one by one cost 14 GB a frame in run 11, before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering 14-90 frames
        // into the opening cutscene; with every frame it ran on). The pass
        // costs some 30 ms a frame this way.
        let mut planned = spans
            .values()
            .map(|(shadow, from, to)| (shadow.clone(), *from, *to, true))
            .collect::<Vec<_>>();
        planned.extend(
            newly_ready
                .into_iter()
                .map(|(shadow, from, to)| (shadow, from, to, false)),
        );
        let ready_copies = planned
            .iter()
            .map(|(shadow, _, _, _)| shadow.id())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let mut refreshes = Vec::new();
        let recording = list.buffer_recording_id();
        {
            let mut coverage = budget.lock().unwrap_or_else(|e| e.into_inner());
            for (shadow, from, to, full) in planned {
                let ranges = if shadow.unwatched() {
                    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                    let interval = *EVERY.get_or_init(|| {
                        std::env::var("VICTORY_UNWATCHED_FRAMES")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(1)
                    });
                    if full && !coverage.ready.allow_full(&shadow, presented, interval) {
                        continue;
                    }
                    coverage
                        .ready
                        .claim(recording, shadow.id(), presented, from, to)
                } else {
                    vec![(from, to)]
                };
                for (from, to) in ranges {
                    refreshes.push((shadow.clone(), from, to));
                }
            }
        }
        let ready_bytes: u64 = refreshes.iter().map(|(_, from, to)| to - from).sum();
        let ready_ranges = refreshes.len() as u64;
        for (shadow, from, to) in refreshes {
            shadow.refresh(list, from, to - from);
        }
        {
            let mut cache = host.buffer_ranges();
            cache.refresh_us += refreshing.elapsed().as_micros() as u64;
            cache.ready_bytes = cache.ready_bytes.saturating_add(ready_bytes);
            cache.ready_ranges += ready_ranges;
        }
        self.heap_textures(list, heap, host);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = scanned,
                views,
                copies = ready_copies,
                mb = ready_bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        log_buffer_warmup(host);
        Some(gpu)
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        static TIME: std::sync::Mutex<TextureStageBudget> =
            std::sync::Mutex::new(TextureStageBudget::new());
        let limits = TextureStageLimits::configured();
        let Some(stage) =
            TextureStageBudget::begin(&TIME, crate::queue::presented(), limits.shared, limits.span)
        else {
            return;
        };
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let budget_frame = crate::queue::presented();
        if FRAME.swap(budget_frame, Ordering::Relaxed) != budget_frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let room = crate::resource::room();
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        host.texture_slots_remapped(frame);
        let (due, census) = host.texture_slots_due(frame, 4096);
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now
            .saturating_sub(PASSED.swap(now, Ordering::Relaxed))
            .max(1);
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        for (look, (index, write)) in due.into_iter().enumerate() {
            if SPENT.load(Ordering::Relaxed) >= limits.bytes || (look != 0 && stage.exhausted()) {
                break;
            }
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            let mut partly = None;
            if let Some(View::Placed(false, resource, desc)) =
                heap.handle(index).and_then(crate::descriptor::view)
            {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(&resource).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(View::Shader(object, _)) =
                heap.handle(index).and_then(crate::descriptor::view)
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            if !room {
                refused += 1;
                if refused > 256 {
                    break;
                }
                continue;
            }
            let view = heap
                .handle(index)
                .and_then(|handle| host.mirror(index, handle, true));
            let placed = view
                .as_ref()
                .and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills =
                crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if let Some(view) = &view {
                readable(list, sampled.kind, view);
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(
                    why,
                    slot = index,
                    "the host device went while textures picked by index were made ready; last: {text}"
                );
                crate::host::log_removal_context();
                return;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            host.texture_slot_done(index, write, again);
        }
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why(
                    "a stage of the ray-tracing structure builder, which is off".to_owned(),
                );
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        ambient::after_dispatch(list, p, &self.compute, &self.heaps);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(
                text,
                " targets {targets:?} depth {:?} state [{}]",
                described(self.depth),
                p.state
            );
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(
                    text,
                    " scissor [{} {} {} {}]",
                    r.left, r.top, r.right, r.bottom
                );
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage
                .root_cbvs
                .iter()
                .chain(&stage.root_srvs)
                .chain(&stage.root_uavs)
            {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT {
                left: 0,
                top: 0,
                right: 1,
                bottom: 1,
            };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    pub(crate) fn dispatch_mesh_groups(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        self.window = None;
        if x == 0 {
            return;
        }
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let fallback = crate::pipeline::host_pipeline(self.pipeline as This)
            .and_then(|p| p.mesh_fallback.as_ref());
        if fallback.is_some() && !mesh::groups_supported(x, y, z) {
            static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| {
                tracing::warn!(
                    x,
                    y,
                    z,
                    "a mesh dispatch outside the compute fallback's one-dimensional 65535-group form"
                )
            });
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
            return;
        }
        let prepared =
            mesh_test() != 1 && mesh_kept(self.pipeline) && self.prepare_for(list, false, true);
        let ran = prepared
            && match fallback {
                Some(fallback) => {
                    let p = crate::pipeline::host_pipeline(self.pipeline as This)
                        .expect("pipeline lives");
                    let ran = self.dispatch_mesh_fallback(list, p, fallback, x);
                    if ran {
                        DRAWN.fetch_add(1, Ordering::Relaxed);
                    }
                    ran
                }
                None => match list.list.cast::<ID3D12GraphicsCommandList6>() {
                    Ok(l6) => {
                        #[cfg(feature = "diagnostics")]
                        list.note_op(|| self.op_text(format!("mesh {x}"), false));
                        self.mesh_scissor(list, true);
                        // SAFETY: an open list with everything set.
                        unsafe { l6.DispatchMesh(x, y, z) };
                        self.mesh_scissor(list, false);
                        DRAWN.fetch_add(1, Ordering::Relaxed);
                        true
                    }
                    Err(_) => false,
                },
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, host.id(), first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected
                                .binary_search(&k)
                                .ok()
                                .and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading();
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        if kind == MESH
            && p.mesh_fallback.is_some()
            && !mesh::indirect_supported(max, count.is_some())
        {
            why("mesh fallback supports one uncounted record");
            return false;
        }
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let signature_kind = if kind == MESH && p.mesh_fallback.is_some() {
            DISPATCH
        } else {
            kind
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(signature_kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        let windowed = windows()
            && !args.in_place()
            && !counter.as_ref().is_some_and(|(copy, _)| copy.in_place())
            && kind != DISPATCH
            && p.mesh_fallback.is_none()
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| {
                (offset - w.base) % u64::from(w.step) == 0
                    && (offset - w.base) / u64::from(w.step) < u64::from(w.records)
            })
            .map(|w| (w.scratch.clone(), offset - w.base));
        let reused = hit.is_some();
        let (bounded, at) = match hit {
            Some(hit) => hit,
            None => {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (
                                step as u32,
                                (if w.records == 1 { 16 } else { w.records * 2 })
                                    .min(fit)
                                    .max(1),
                            )
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh(
                    list,
                    arguments,
                    len.max(u64::from(step) * u64::from(records)),
                );
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = if kind == MESH && p.mesh_fallback.is_some() {
                    crate::indirect::bound_mesh_fallback(
                        list,
                        (&args.buffer, offset),
                        (kind, step, records),
                    )
                } else if windowed && records > 1 {
                    crate::indirect::bound_ahead(
                        list,
                        (&args.buffer, offset),
                        (kind, step, records),
                    )
                } else {
                    crate::indirect::bound(
                        list,
                        (&args.buffer, offset),
                        count_arg,
                        (kind, step, records),
                    )
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            if !reused {
                crate::indirect::done(list, &bounded);
            }
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        if kind == MESH
            && let Some(fallback) = &p.mesh_fallback
        {
            let ran = self.execute_mesh_fallback(list, p, fallback, &bounded, stride);
            if !ran {
                crate::indirect::done(list, &bounded);
                why("mesh fallback scheduling failed");
                return false;
            }
        } else {
            if kind == MESH {
                self.mesh_scissor(list, true);
            }
            // SAFETY: an open list with everything set; the scratch buffer and
            // host copies are kept alive by the list, and the signature is cached.
            unsafe {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &bounded,
                    at,
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                )
            };
            if kind == MESH {
                self.mesh_scissor(list, false);
            }
        }
        crate::indirect::done(list, &bounded);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x} of {r:?}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

fn host_topology(topology: u32) -> u32 {
    match topology {
        RECTANGLES => 4,
        TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
            once!(topology, "console topology drawn as a triangle list");
            4
        }
        t => t,
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("STENCIL REF {reference:#x}")
    });
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer_test_limits() -> BufferWarmLimits {
        BufferWarmLimits {
            span: std::time::Duration::from_secs(60),
            bytes: u64::MAX,
        }
    }

    fn buffer_test_write(memory: &crate::resource::Memory, offset: usize, len: usize, value: u8) {
        let address = memory.address() as usize + offset;
        crate::watch::writing(address, len);
        unsafe { std::ptr::write_bytes(address as *mut u8, value, len) };
        crate::watch::wrote(address, len);
    }

    fn buffer_test_byte(shadow: &crate::shadow::Shadow, offset: usize) -> u8 {
        let mut mapped = std::ptr::null_mut();
        unsafe {
            shadow.buffer.Map(0, None, Some(&mut mapped)).unwrap();
            let value = *(mapped as *const u8).add(offset);
            shadow.buffer.Unmap(0, None);
            value
        }
    }

    #[test]
    fn buffer_ready_coverage_matches_a_byte_oracle_and_resets_on_present() {
        let mut coverage = BufferReadyCoverage::default();
        let mut bytes = [false; 128];
        let mut seed = 0x917b_2841u32;
        for _ in 0..2048 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let a = seed as usize % 129;
            let b = (seed >> 16) as usize % 129;
            let (from, to) = (a.min(b), a.max(b));
            let mut expected = Vec::new();
            let mut at = from;
            while at < to {
                if bytes[at] {
                    at += 1;
                    continue;
                }
                let first = at;
                while at < to && !bytes[at] {
                    at += 1;
                }
                expected.push((first as u64, at as u64));
            }
            assert_eq!(coverage.claim(1, 1, 10, from as u64, to as u64), expected);
            bytes[from..to].fill(true);
        }
        assert_eq!(coverage.claim(1, 2, 10, 0, 128), [(0, 128)]);
        assert_eq!(coverage.claim(2, 1, 10, 0, 128), [(0, 128)]);
        assert_eq!(coverage.claim(1, 1, 11, 0, 128), [(0, 128)]);
        assert_eq!(coverage.claim(1, 1, 9, 0, 128), [(0, 128)]);
        assert!(coverage.claim(1, 1, 11, 0, 128).is_empty());
    }

    #[test]
    fn buffer_warm_cursor_survives_rescans_and_changed_slot_backlogs() {
        let mut work = crate::descriptor::host::BufferWork::default();
        let slots: std::collections::BTreeSet<_> = (0..1000u32).collect();
        let after = |cursor| {
            use std::ops::Bound::{Excluded, Unbounded};
            match cursor {
                Some(index) => slots.range((Excluded(index), Unbounded)).next().copied(),
                None => slots.first().copied(),
            }
        };
        work.scan();
        for index in 0..10 {
            assert_eq!(work.next(after), Some(index));
        }
        assert_eq!(work.cursor(), Some(9));
        work.enqueue(3);
        work.enqueue(3);
        assert_eq!(work.pending(), 1);
        work.scan();
        assert_eq!(work.cursor(), Some(9));
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..3000 {
            let Some(index) = work.next(after) else { break };
            seen.insert(index);
        }
        assert_eq!(seen, slots);
        assert!(!work.has_work());
    }

    #[test]
    fn buffer_warm_budget_shares_atomic_overruns_and_rotates_heap_priority() {
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let a = std::sync::Arc::new(());
        let b = std::sync::Arc::new(());
        let limits = BufferWarmLimits {
            span: std::time::Duration::from_millis(1),
            bytes: u64::MAX,
        };
        for frame in 0..8 {
            let first = if frame % 2 == 0 { 10 } else { 20 };
            let mut served = Vec::new();
            for (heap, life) in [(10, &a), (20, &b)] {
                if let Some(mut permit) =
                    BufferWarmBudget::begin(&budget, heap, life, frame, limits)
                {
                    permit.began = std::time::Instant::now() - std::time::Duration::from_millis(2);
                    served.push(heap);
                }
            }
            assert_eq!(served, [first], "frame {frame}");
            assert!(budget.lock().unwrap().elapsed >= limits.span);
        }
        drop(b);
        assert!(BufferWarmBudget::begin(&budget, 10, &a, 8, limits).is_some());
    }

    #[test]
    fn buffer_warm_budget_allows_only_one_concurrent_atomic_job() {
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let life = std::sync::Arc::new(());
        let barrier = std::sync::Barrier::new(8);
        let granted = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        let permit =
                            BufferWarmBudget::begin(&budget, 1, &life, 0, buffer_test_limits());
                        let granted = permit.is_some();
                        barrier.wait();
                        drop(permit);
                        granted
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|job| usize::from(job.join().unwrap()))
                .sum::<usize>()
        });
        assert_eq!(granted, 1);
    }

    struct TestBindlessHeap {
        object: This,
        address: u64,
        resource: usize,
        shadow: std::sync::Arc<crate::shadow::Shadow>,
    }

    impl TestBindlessHeap {
        fn new(address: u64, resource: usize) -> Self {
            Self::with_slots(address, resource, 1)
        }

        fn with_slots(address: u64, resource: usize, count: u32) -> Self {
            let desc = [0u32, count, 1, 0, 0, 0, 0, 0];
            let iid = crate::object::guid(
                0x8efb471d,
                0x616c,
                0x4f49,
                [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
            );
            let mut object = std::ptr::null();
            assert_eq!(
                crate::descriptor::create_heap(
                    std::ptr::null(),
                    desc.as_ptr().cast(),
                    &iid,
                    &mut object,
                ),
                crate::object::S_OK,
            );
            crate::shadow::register(resource, address, 65536);
            let shadow = crate::shadow::find(address, 65536).expect("registered buffer copy");
            let heap = Self {
                object,
                address,
                resource,
                shadow,
            };
            let buffer = [1u32, 0, 0, 0, 65536, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mut view = [0u32; 16];
            view[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS.0 as u32;
            view[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
            view[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view[6] = 64;
            view[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
            view[10] = address as u32;
            view[11] = (address >> 32) as u32;
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &buffer,
                &view,
                crate::descriptor::heap(object).handle(0).unwrap(),
            );
            heap
        }

        fn host(&self) -> &crate::descriptor::host::HostHeap {
            crate::descriptor::heap(self.object)
                .host()
                .expect("shader-visible host heap")
        }

        fn rewrite_buffer(&self, address: u64) {
            self.rewrite_buffer_slot(0, address, 0);
        }

        fn rewrite_buffer_slot(&self, index: u32, address: u64, first_element: u32) {
            let buffer = [1u32, 0, 0, 0, 65536, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mut view = [0u32; 16];
            view[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS.0 as u32;
            view[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
            view[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view[4] = first_element;
            view[6] = 64;
            view[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
            view[10] = address as u32;
            view[11] = (address >> 32) as u32;
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &buffer,
                &view,
                crate::descriptor::heap(self.object).handle(index).unwrap(),
            );
        }
    }

    impl Drop for TestBindlessHeap {
        fn drop(&mut self) {
            crate::shadow::unregister(self.resource, self.address);
            crate::object::slot::<extern "system" fn(This) -> u32>(self.object, 0x10)(self.object);
        }
    }

    struct TestObjectBuffer(This);

    impl TestObjectBuffer {
        fn at(address: u64) -> Self {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 65536,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let iid = crate::object::guid(
                0x696442be,
                0xa72e,
                0x4059,
                [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
            );
            let mut object = std::ptr::null();
            assert_eq!(
                crate::resource::create_at_address(
                    std::ptr::null(),
                    address,
                    std::ptr::from_ref(&desc).cast(),
                    0,
                    std::ptr::null(),
                    &iid,
                    &mut object
                ),
                crate::object::S_OK
            );
            Self(object)
        }

        fn release(&mut self) {
            if !self.0.is_null() {
                crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
                self.0 = std::ptr::null();
            }
        }
    }

    impl Drop for TestObjectBuffer {
        fn drop(&mut self) {
            self.release();
        }
    }

    fn buffer_test_object_view(heap: &TestBindlessHeap, index: u32, resource: &TestObjectBuffer) {
        let mut view = [0u32; 10];
        view[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS.0 as u32;
        view[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        view[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view[6] = 64;
        view[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        crate::descriptor::create_srv(
            std::ptr::null(),
            resource.0,
            &view,
            crate::descriptor::heap(heap.object).handle(index).unwrap(),
        );
    }

    #[test]
    fn buffer_ready_release_reap_frees_stale_refs_with_no_cold_budget_and_keeps_live_owners() {
        let accounting = crate::shadow::TestCopyBudget::new(8 * 65536);
        let old_memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        let live_memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        let list_memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        let solo_memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        let new_memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        let mut old = TestObjectBuffer::at(old_memory.address());
        let live = TestObjectBuffer::at(live_memory.address());
        let mut list_owned = TestObjectBuffer::at(list_memory.address());
        let mut solo = TestObjectBuffer::at(solo_memory.address());
        let new_generation = TestObjectBuffer::at(new_memory.address());
        let heap = TestBindlessHeap::with_slots(0x7edb_6100_0000, usize::MAX - 0x6f30, 7);
        buffer_test_object_view(&heap, 1, &old);
        heap.rewrite_buffer_slot(2, old_memory.address(), 64);
        buffer_test_object_view(&heap, 3, &live);
        buffer_test_object_view(&heap, 4, &old);
        buffer_test_object_view(&heap, 5, &list_owned);
        buffer_test_object_view(&heap, 6, &solo);
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let mut state = State::default();
        state.set_heaps(&[heap.object]);
        let first = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(&first, 1, 200, &budget, buffer_test_limits())
            .unwrap();
        assert_eq!(heap.host().buffer_ranges().ranges.len(), 7);
        let keep = host.new_list().unwrap();
        let (old_weak, list_weak, solo_weak) = {
            let cache = heap.host().buffer_ranges();
            assert!(std::sync::Arc::ptr_eq(
                &cache.ranges[&1].0,
                &cache.ranges[&2].0
            ));
            cache.ranges[&5]
                .0
                .refresh(&keep, list_memory.address(), 256);
            (
                std::sync::Arc::downgrade(&cache.ranges[&1].0),
                std::sync::Arc::downgrade(&cache.ranges[&5].0),
                std::sync::Arc::downgrade(&cache.ranges[&6].0),
            )
        };
        first.close();
        first.reset();
        old.release();
        list_owned.release();
        solo.release();
        buffer_test_object_view(&heap, 4, &new_generation);
        let handle = crate::descriptor::heap(heap.object).handle(4).unwrap();
        heap.host().mirror_at_present(4, handle, true, 201).unwrap();
        let new_shadow = crate::shadow::find_for_view(new_memory.address(), 65536).unwrap();
        let references = crate::host::references(&new_shadow.buffer);
        let before = accounting.used();
        let blocked = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(
                &blocked,
                2,
                201,
                &budget,
                BufferWarmLimits {
                    span: std::time::Duration::ZERO,
                    bytes: 0,
                },
            )
            .unwrap();
        {
            let cache = heap.host().buffer_ranges();
            assert_eq!(cache.stale_slots, 4);
            assert_eq!(cache.stale_host_slots, 3);
            assert_eq!(cache.stale_shadow_bytes, 3 * 65536);
            for index in [0, 2, 3] {
                assert!(cache.ranges.contains_key(&index));
            }
            for index in [1, 4, 5, 6] {
                assert!(!cache.ranges.contains_key(&index));
            }
        }
        assert!(!heap.host().buffer_slot_ready(1));
        assert!(heap.host().buffer_slot_ready(2));
        assert!(heap.host().buffer_slot_ready(3));
        assert!(heap.host().buffer_slot_ready(4));
        assert!(!heap.host().buffer_slot_ready(5));
        assert!(!heap.host().buffer_slot_ready(6));
        assert_eq!(crate::host::references(&new_shadow.buffer), references);
        assert!(old_weak.upgrade().is_some());
        assert!(list_weak.upgrade().is_some());
        assert!(solo_weak.upgrade().is_none());
        assert_eq!(accounting.used(), before - 65536);
        keep.close();
        keep.reset();
        assert!(list_weak.upgrade().is_none());
        assert_eq!(accounting.used(), before - 2 * 65536);
        assert!(old_weak.upgrade().is_some());
    }

    #[test]
    fn buffer_ready_release_reap_skips_an_epoch_already_claimed_by_another_recorder() {
        let memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        let mut object = TestObjectBuffer::at(memory.address());
        let heap = TestBindlessHeap::with_slots(0x7edb_6200_0000, usize::MAX - 0x6f31, 2);
        buffer_test_object_view(&heap, 1, &object);
        let host = crate::host::host().unwrap();
        let list = host.new_list().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let mut state = State::default();
        state.set_heaps(&[heap.object]);
        state
            .heap_buffers_budgeted(&list, 1, 300, &budget, buffer_test_limits())
            .unwrap();
        list.close();
        list.reset();
        object.release();
        let title_heap = crate::descriptor::heap(heap.object);
        let passes = heap.host().buffer_ranges().reap_passes;
        heap.host().buffer_ranges().reaping = true;
        let heap_address = heap.object as usize;
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        let title = crate::descriptor::heap(heap_address as This);
                        reap_buffer_ranges(title, title.host().unwrap());
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
        });
        assert_eq!(heap.host().buffer_ranges().reap_passes, passes);
        assert!(heap.host().buffer_ranges().ranges.contains_key(&1));
        {
            let _pass = BufferReapPass(heap.host());
        }
        assert!(!heap.host().buffer_ranges().reaping);
        reap_buffer_ranges(title_heap, heap.host());
        assert_eq!(heap.host().buffer_ranges().reap_passes, passes + 1);
        assert!(!heap.host().buffer_ranges().ranges.contains_key(&1));
        reap_buffer_ranges(title_heap, heap.host());
        assert_eq!(heap.host().buffer_ranges().reap_passes, passes + 1);
    }

    #[test]
    fn buffer_ready_batches_follow_present_and_targeted_refresh_remains_immediate() {
        let memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        buffer_test_write(&memory, 0, 256, 0x11);
        let heap = TestBindlessHeap::new(memory.address(), usize::MAX - 0x6f20);
        heap.shadow.update(memory.address() as usize, 256);
        assert!(heap.shadow.unwatched());
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let blocked = BufferWarmLimits {
            span: std::time::Duration::ZERO,
            bytes: 0,
        };
        let mut state = State::default();
        state.set_heaps(&[heap.object]);
        let first = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(&first, 1, 100, &budget, buffer_test_limits())
            .unwrap();
        first.close();
        host.execute(&[&first]);
        assert_eq!(buffer_test_byte(&heap.shadow, 0), 0x11);
        assert_eq!(heap.host().buffer_ranges().full_ready_batches, 1);
        assert_eq!(heap.host().buffer_ranges().ready_bytes, 256);
        buffer_test_write(&memory, 0, 256, 0xa7);
        let same = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(&same, 2, 100, &budget, blocked)
            .unwrap();
        same.close();
        host.execute(&[&same]);
        assert_eq!(heap.host().buffer_ranges().full_ready_batches, 1);
        assert_eq!(heap.host().buffer_ranges().ready_bytes, 256);
        assert_eq!(buffer_test_byte(&heap.shadow, 0), 0x11);
        let next = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(&next, 3, 101, &budget, blocked)
            .unwrap();
        next.close();
        host.execute(&[&next]);
        assert_eq!(heap.host().buffer_ranges().full_ready_batches, 2);
        assert_eq!(heap.host().buffer_ranges().ready_bytes, 512);
        assert_eq!(buffer_test_byte(&heap.shadow, 0), 0xa7);
        buffer_test_write(&memory, 0, 256, 0xc3);
        let targeted = host.new_list().unwrap();
        let handle = crate::descriptor::heap(heap.object).handle(0).unwrap();
        let view = heap.host().mirror_at_present(0, handle, true, 101).unwrap();
        let (shadow, address, len) = crate::descriptor::host::buffer_copy(&view).unwrap();
        shadow.refresh(&targeted, address, len);
        state
            .heap_buffers_budgeted(&targeted, 4, 101, &budget, blocked)
            .unwrap();
        targeted.close();
        host.execute(&[&targeted]);
        assert_eq!(buffer_test_byte(&heap.shadow, 0), 0xc3);
        assert_eq!(heap.host().buffer_ranges().full_ready_batches, 2);
        assert_eq!(heap.host().buffer_ranges().ready_bytes, 512);
    }

    #[test]
    fn buffer_ready_coverage_is_shared_by_heaps_and_copies_only_new_subranges() {
        let memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        buffer_test_write(&memory, 0, 768, 0x11);
        let a = TestBindlessHeap::new(memory.address(), usize::MAX - 0x6f21);
        let b = TestBindlessHeap::with_slots(memory.address(), usize::MAX - 0x6f22, 2);
        assert!(std::sync::Arc::ptr_eq(&a.shadow, &b.shadow));
        a.shadow.update(memory.address() as usize, 256);
        assert!(a.shadow.unwatched());
        b.rewrite_buffer_slot(0, memory.address(), 32);
        b.rewrite_buffer_slot(1, memory.address(), 128);
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let mut state = State::default();
        state.set_heaps(&[a.object]);
        let first = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(&first, 1, 100, &budget, buffer_test_limits())
            .unwrap();
        buffer_test_write(&memory, 128, 256, 0xa7);
        buffer_test_write(&memory, 512, 256, 0xc3);
        state.set_heaps(&[b.object]);
        state
            .heap_buffers_budgeted(&first, 2, 100, &budget, buffer_test_limits())
            .unwrap();
        first.close();
        host.execute(&[&first]);
        assert_eq!(a.host().buffer_ranges().ready_bytes, 256);
        assert_eq!(b.host().buffer_ranges().full_ready_batches, 1);
        assert_eq!(b.host().buffer_ranges().ready_bytes, 384);
        assert_eq!(b.host().buffer_ranges().ready_ranges, 2);
        assert_eq!(buffer_test_byte(&a.shadow, 128), 0xa7);
        assert_eq!(buffer_test_byte(&a.shadow, 256), 0xa7);
        assert_eq!(buffer_test_byte(&a.shadow, 512), 0xc3);
        let mut coverage = BufferReadyCoverage::default();
        assert!(coverage.allow_full(&a.shadow, 10, 3));
        assert!(coverage.allow_full(&a.shadow, 10, 3));
        assert!(!coverage.allow_full(&a.shadow, 11, 3));
        assert_eq!(
            coverage.claim(1, a.shadow.id(), 11, 1024, 1280),
            [(1024, 1280)]
        );
        assert!(coverage.allow_full(&a.shadow, 13, 3));
    }

    #[test]
    fn buffer_ready_independent_new_alias_refreshes_when_the_first_list_never_executes() {
        let memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        buffer_test_write(&memory, 0, 256, 0x11);
        let a = TestBindlessHeap::new(memory.address(), usize::MAX - 0x6f23);
        let b = TestBindlessHeap::new(memory.address(), usize::MAX - 0x6f24);
        assert!(std::sync::Arc::ptr_eq(&a.shadow, &b.shadow));
        a.shadow.update(memory.address() as usize, 256);
        assert!(a.shadow.unwatched());
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let first = host.new_list().unwrap();
        let second = host.new_list().unwrap();
        let mut state = State::default();
        buffer_test_write(&memory, 0, 256, 0xa7);
        state.set_heaps(&[a.object]);
        state
            .heap_buffers_budgeted(&first, 1, 100, &budget, buffer_test_limits())
            .unwrap();
        state.set_heaps(&[b.object]);
        state
            .heap_buffers_budgeted(&second, 2, 100, &budget, buffer_test_limits())
            .unwrap();
        assert_ne!(first.buffer_recording_id(), second.buffer_recording_id());
        assert_eq!(buffer_test_byte(&a.shadow, 0), 0x11);
        second.close();
        host.execute(&[&second]);
        assert_eq!(buffer_test_byte(&a.shadow, 0), 0xa7);
        assert_eq!(b.host().buffer_ranges().ready_bytes, 256);
        let old_recording = first.buffer_recording_id();
        first.close();
        first.reset();
        assert_ne!(first.buffer_recording_id(), old_recording);
        let c = TestBindlessHeap::new(memory.address(), usize::MAX - 0x6f25);
        assert!(std::sync::Arc::ptr_eq(&a.shadow, &c.shadow));
        buffer_test_write(&memory, 0, 256, 0xc3);
        state.set_heaps(&[c.object]);
        state
            .heap_buffers_budgeted(&first, 3, 100, &budget, buffer_test_limits())
            .unwrap();
        assert_eq!(c.host().buffer_ranges().ready_bytes, 256);
        first.close();
        host.execute(&[&first]);
        assert_eq!(buffer_test_byte(&a.shadow, 0), 0xc3);
    }

    #[test]
    fn buffer_warm_byte_budget_counts_unique_real_copies_and_one_oversized_job() {
        let a = TestBindlessHeap::new(0x7edb_5100_0000, usize::MAX - 0x6ef4);
        let b = TestBindlessHeap::new(0x7edb_5101_0000, usize::MAX - 0x6ef5);
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let life = std::sync::Arc::new(());
        let limits = BufferWarmLimits {
            span: std::time::Duration::from_secs(60),
            bytes: 2 * 65536,
        };
        for (shadow, expected) in [(&a.shadow, 65536), (&a.shadow, 0), (&b.shadow, 65536)] {
            let mut permit = BufferWarmBudget::begin(&budget, 1, &life, 0, limits).unwrap();
            assert_eq!(permit.copied(shadow), expected);
        }
        assert_eq!(budget.lock().unwrap().bytes, 2 * 65536);
        assert!(BufferWarmBudget::begin(&budget, 1, &life, 0, limits).is_none());
        let oversized = std::sync::Mutex::new(BufferWarmBudget::default());
        let limits = BufferWarmLimits { bytes: 1, ..limits };
        {
            let mut permit = BufferWarmBudget::begin(&oversized, 1, &life, 0, limits).unwrap();
            permit.copied(&a.shadow);
        }
        assert!(BufferWarmBudget::begin(&oversized, 1, &life, 0, limits).is_none());
    }

    #[test]
    fn buffer_warm_pending_heap_does_not_stop_ready_buffer_refresh_or_targeted_views() {
        let memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        let a = TestBindlessHeap::new(memory.address(), usize::MAX - 0x6ef6);
        let b = TestBindlessHeap::new(0x7edb_5201_0000, usize::MAX - 0x6ef7);
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let first = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[a.object]);
        state
            .heap_buffers_budgeted(&first, 1, 1, &budget, buffer_test_limits())
            .unwrap();
        first.close();
        host.execute(&[&first]);
        let blocked = BufferWarmLimits {
            span: std::time::Duration::ZERO,
            bytes: 0,
        };
        let second = host.new_list().unwrap();
        crate::watch::writing(memory.address() as usize, 256);
        unsafe { std::ptr::write_bytes(memory.address() as *mut u8, 0xa7, 256) };
        crate::watch::wrote(memory.address() as usize, 256);
        state
            .heap_buffers_budgeted(&second, 2, 2, &budget, blocked)
            .unwrap();
        state.set_heaps(&[b.object]);
        state
            .heap_buffers_budgeted(&second, 2, 2, &budget, blocked)
            .unwrap();
        assert_eq!(a.host().buffer_ranges().ranges.len(), 1);
        assert!(b.host().buffer_ranges().ranges.is_empty());
        assert!(b.host().buffer_ranges().work.has_work());
        let before = crate::host::references(&b.shadow.buffer);
        b.host()
            .mirror_at_present(
                0,
                crate::descriptor::heap(b.object).handle(0).unwrap(),
                true,
                2,
            )
            .unwrap();
        assert_eq!(crate::host::references(&b.shadow.buffer), before + 1);
        second.close();
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        unsafe {
            a.shadow.buffer.Map(0, None, Some(&mut mapped)).unwrap();
            assert_eq!(*(mapped as *const u8), 0xa7);
            a.shadow.buffer.Unmap(0, None);
        }
    }

    #[test]
    fn buffer_warm_budget_retry_survives_rescan_and_recovers_on_next_present() {
        let heap = TestBindlessHeap::new(0x7edb_5300_0000, usize::MAX - 0x6ef8);
        const NEW: u64 = 0x7edb_5301_0000;
        const RESOURCE: usize = usize::MAX - 0x6ef9;
        let admission = crate::shadow::TestCopyBudget::new(65536);
        const FILLER: u64 = 0x7edb_5302_0000;
        const FILLER_RESOURCE: usize = usize::MAX - 0x6efa;
        crate::shadow::register(FILLER_RESOURCE, FILLER, 65536);
        let filler = crate::shadow::find(FILLER, 65536).unwrap();
        assert_eq!(admission.used(), 65536);
        crate::shadow::register(RESOURCE, NEW, 65536);
        heap.rewrite_buffer(NEW);
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let mut state = State::default();
        state.set_heaps(&[heap.object]);
        let first = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(&first, 10, 10, &budget, buffer_test_limits())
            .unwrap();
        assert!(heap.host().buffer_retry_pending(0));
        assert!(heap.host().buffer_ranges().ranges.is_empty());
        heap.host().buffer_ranges().work.scan();
        crate::shadow::unregister(FILLER_RESOURCE, FILLER);
        drop(filler);
        assert_eq!(admission.used(), 0);
        let second = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(&second, 11, 11, &budget, buffer_test_limits())
            .unwrap();
        assert!(!heap.host().buffer_retry_pending(0));
        assert_eq!(heap.host().buffer_ranges().ranges.len(), 1);
        assert_eq!(admission.used(), 65536);
        crate::shadow::unregister(RESOURCE, NEW);
    }

    #[test]
    fn buffer_warm_permanent_invalid_view_never_becomes_ready_or_pinned() {
        let heap = TestBindlessHeap::new(0x7edb_5400_0000, usize::MAX - 0x6efb);
        let resource = [1u32, 0, 0, 0, 65536, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut view = [0u32; 16];
        view[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        view[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view[6] = 64;
        view[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        view[10] = heap.address as u32;
        view[11] = (heap.address >> 32) as u32;
        crate::descriptor::create_placed_srv(
            std::ptr::null(),
            &resource,
            &view,
            crate::descriptor::heap(heap.object).handle(0).unwrap(),
        );
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let mut state = State::default();
        state.set_heaps(&[heap.object]);
        let references = crate::host::references(&heap.shadow.buffer);
        for frame in 1..=2 {
            heap.host().buffer_ranges().work.scan();
            let list = host.new_list().unwrap();
            state
                .heap_buffers_budgeted(&list, frame, frame, &budget, buffer_test_limits())
                .unwrap();
            assert!(heap.host().buffer_ranges().ranges.is_empty());
            assert!(!heap.host().buffer_retry_pending(0));
            assert!(!heap.host().buffer_slot_ready(0));
            assert_eq!(crate::host::references(&heap.shadow.buffer), references);
            assert_eq!(std::sync::Arc::strong_count(&heap.shadow), 2);
        }
    }

    #[test]
    fn buffer_warm_new_span_bypasses_unwatched_suppression_in_the_same_frame() {
        let memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
        unsafe { std::ptr::write_bytes(memory.address() as *mut u8, 0x11, 512) };
        let heap = TestBindlessHeap::with_slots(memory.address(), usize::MAX - 0x6efc, 2);
        heap.rewrite_buffer_slot(1, memory.address(), 64);
        let host = crate::host::host().unwrap();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        let mut state = State::default();
        state.set_heaps(&[heap.object]);
        heap.shadow.update(memory.address() as usize, 256);
        assert!(heap.shadow.unwatched());
        let first = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(
                &first,
                7,
                7,
                &budget,
                BufferWarmLimits {
                    bytes: 1,
                    ..buffer_test_limits()
                },
            )
            .unwrap();
        assert_eq!(heap.host().buffer_ranges().ranges.len(), 1);
        first.close();
        host.execute(&[&first]);
        assert!(heap.shadow.unwatched());
        crate::watch::writing(memory.address() as usize + 256, 256);
        unsafe { std::ptr::write_bytes((memory.address() + 256) as *mut u8, 0xa7, 256) };
        crate::watch::wrote(memory.address() as usize + 256, 256);
        let second = host.new_list().unwrap();
        state
            .heap_buffers_budgeted(
                &second,
                7,
                7,
                &budget,
                BufferWarmLimits {
                    bytes: 131072,
                    ..buffer_test_limits()
                },
            )
            .unwrap();
        assert_eq!(heap.host().buffer_ranges().ranges.len(), 2);
        second.close();
        host.execute(&[&second]);
        let mut mapped = std::ptr::null_mut();
        unsafe {
            heap.shadow.buffer.Map(0, None, Some(&mut mapped)).unwrap();
            assert_eq!(*(mapped as *const u8), 0x11);
            assert_eq!(*(mapped as *const u8).add(256), 0xa7);
            heap.shadow.buffer.Unmap(0, None);
        }
    }

    #[test]
    fn bindless_buffer_heaps_switch_independently_in_one_recorded_list() {
        const FRAME: u64 = 0x7edb_3000;
        let first = TestBindlessHeap::new(0x7edb_3000_0000, usize::MAX - 0x6ef0);
        let second = TestBindlessHeap::new(0x7edb_3001_0000, usize::MAX - 0x6ef1);
        let host = crate::host::host().expect("D3D12 host");
        let list = host.new_list().expect("recorded list");
        let strides = |heap: &TestBindlessHeap| {
            crate::host::references(heap.host().strides_buffer().unwrap())
        };
        let before = [strides(&first), strides(&second)];
        let views = [
            crate::host::references(&first.shadow.buffer),
            crate::host::references(&second.shadow.buffer),
        ];
        let mut state = State::default();
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        for heap in [&first, &second, &first] {
            state.set_heaps(&[heap.object]);
            assert_eq!(
                state.heap_buffers_budgeted(&list, FRAME, FRAME, &budget, buffer_test_limits()),
                heap.host().strides_gpu()
            );
        }
        for (n, heap) in [&first, &second].into_iter().enumerate() {
            let cache = heap.host().buffer_ranges();
            assert_eq!(cache.passes, 1);
            assert_eq!(cache.last_frame, Some(FRAME));
            assert!(cache.ranges.contains_key(&0));
            assert_eq!(strides(heap), before[n] + 1);
            assert_eq!(crate::host::references(&heap.shadow.buffer), views[n] + 1);
            assert_eq!(std::sync::Arc::strong_count(&heap.shadow), 4);
        }
        list.close();
        list.reset();
        for (n, heap) in [&first, &second].into_iter().enumerate() {
            assert_eq!(strides(heap), before[n]);
            state.set_heaps(&[heap.object]);
            state
                .heap_buffers_budgeted(&list, FRAME, FRAME, &budget, buffer_test_limits())
                .unwrap();
            assert_eq!(strides(heap), before[n] + 1);
            assert_eq!(heap.host().buffer_ranges().passes, 1);
        }
    }

    #[test]
    fn bindless_buffer_heaps_prepare_independently_in_one_frame_across_lists() {
        const FRAME: u64 = 0x7edb_4000;
        let first = TestBindlessHeap::new(0x7edb_4000_0000, usize::MAX - 0x6ef2);
        let second = TestBindlessHeap::new(0x7edb_4001_0000, usize::MAX - 0x6ef3);
        let host = crate::host::host().expect("D3D12 host");
        let a = host.new_list().expect("first recorded list");
        let b = host.new_list().expect("second recorded list");
        let budget = std::sync::Mutex::new(BufferWarmBudget::default());
        for (heap, list) in [(&first, &a), (&second, &b)] {
            let stride_before = crate::host::references(heap.host().strides_buffer().unwrap());
            let view_before = crate::host::references(&heap.shadow.buffer);
            let mut state = State::default();
            state.set_heaps(&[heap.object]);
            assert_eq!(
                state.heap_buffers_budgeted(list, FRAME, FRAME, &budget, buffer_test_limits()),
                heap.host().strides_gpu()
            );
            let cache = heap.host().buffer_ranges();
            assert_eq!(cache.passes, 1);
            assert_eq!(cache.last_frame, Some(FRAME));
            assert!(cache.ranges.contains_key(&0));
            assert_eq!(
                crate::host::references(heap.host().strides_buffer().unwrap()),
                stride_before + 1
            );
            assert_eq!(
                crate::host::references(&heap.shadow.buffer),
                view_before + 1
            );
            assert_eq!(std::sync::Arc::strong_count(&heap.shadow), 4);
        }
    }
    #[test]
    fn completed_dynamic_tables_require_a_new_recording_before_resubmission() {
        let mut state = State::default();
        state.tables_done();
        assert!(!state.tables_expired());
        state.dynamic.push((0, 1, 4, 1));
        state.tables_gone = false;
        state.tables_done();
        assert!(state.tables_expired());
        state.tables_done();
        assert!(state.tables_expired());
        state = State::default();
        assert!(!state.tables_expired());
    }

    #[test]
    fn texture_stage_budget_is_shared_and_does_not_wait_for_another_recorder() {
        let budget = std::sync::Mutex::new(TextureStageBudget::new());
        let span = std::time::Duration::from_secs(1);
        let first = TextureStageBudget::begin(&budget, 10, true, span).unwrap();
        std::thread::scope(|scope| {
            assert!(
                scope
                    .spawn(|| TextureStageBudget::begin(&budget, 10, true, span).is_none())
                    .join()
                    .unwrap()
            );
        });
        drop(first);
        let next = TextureStageBudget::begin(&budget, 10, true, span).unwrap();
        assert!(next.left < span);
        drop(next);
        let fresh = TextureStageBudget::begin(&budget, 11, true, span).unwrap();
        assert_eq!(fresh.left, span);
        drop(fresh);
        assert!(TextureStageBudget::begin(&budget, 10, true, span).is_none());
    }

    #[test]
    fn texture_stage_overrun_exhausts_the_frame_even_on_early_return() {
        let budget = std::sync::Mutex::new(TextureStageBudget::new());
        let span = std::time::Duration::from_millis(1);
        let work = || {
            let pass = TextureStageBudget::begin(&budget, 20, true, span).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(3));
            assert!(pass.exhausted());
        };
        work();
        assert!(TextureStageBudget::begin(&budget, 20, true, span).is_none());
        assert!(TextureStageBudget::begin(&budget, 21, true, span).is_some());
        assert!(TextureStageBudget::begin(&budget, 21, true, std::time::Duration::ZERO).is_none());
    }

    struct TestTexture(This);

    impl TestTexture {
        fn new(width: u64, height: u32, format: i32, flags: i32) -> Self {
            let properties = [1u32, 0, 0, 1, 1];
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width,
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT(format),
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAGS(flags),
                ..Default::default()
            };
            let iid = crate::object::guid(
                0x696442be,
                0xa72e,
                0x4059,
                [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
            );
            let mut resource = std::ptr::null();
            assert_eq!(
                crate::resource::create_committed(
                    std::ptr::null(),
                    std::ptr::from_ref(&properties).cast(),
                    0,
                    std::ptr::from_ref(&desc).cast(),
                    0,
                    std::ptr::null(),
                    &iid,
                    &mut resource,
                ),
                crate::object::S_OK,
            );
            Self(resource)
        }
    }

    impl Drop for TestTexture {
        fn drop(&mut self) {
            unsafe {
                let table = *self.0.cast::<*const *const ()>();
                let release: extern "system" fn(This) -> u32 = std::mem::transmute(*table.add(2));
                release(self.0);
            }
        }
    }

    #[test]
    fn unused_depth_view_is_deferred_and_rebound_when_pipeline_usage_changes() {
        let Some(host) = crate::host::host() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        let large = TestTexture::new(1920, 1080, 28, 1);
        let small = TestTexture::new(1280, 720, 28, 1);
        let depth = TestTexture::new(1280, 720, 40, 2);
        let replacement_depth = TestTexture::new(1280, 720, 40, 2);
        let mut large_view = [0u64; 14];
        let mut small_view = [0u64; 14];
        let mut depth_view = [0u64; 14];
        let large_handle = large_view.as_mut_ptr() as u64;
        let small_handle = small_view.as_mut_ptr() as u64;
        let depth_handle = depth_view.as_mut_ptr() as u64;
        crate::descriptor::create_rtv(std::ptr::null(), large.0, std::ptr::null(), large_handle);
        crate::descriptor::create_rtv(std::ptr::null(), small.0, std::ptr::null(), small_handle);
        crate::descriptor::create_dsv(std::ptr::null(), depth.0, std::ptr::null(), depth_handle);
        let list = host.new_list().expect("host list");
        let mut state = State::default();
        state.set_targets(&list, &[large_handle], depth_handle);
        crate::descriptor::create_rtv(std::ptr::null(), small.0, std::ptr::null(), large_handle);
        assert_eq!(state.host_targets, None);
        assert_eq!(state.depth, 0);
        state.bind_targets(&list, false);
        assert_eq!(state.targets, [large.0 as usize]);
        assert_eq!(state.host_targets, Some(false));
        assert_eq!(state.depth, 0);
        unsafe { host.device.GetDeviceRemovedReason() }.expect("unused mismatched DSV was bound");

        state.set_targets(&list, &[small_handle], depth_handle);
        state.bind_targets(&list, true);
        assert_eq!(state.depth, depth.0 as usize);
        assert_eq!(state.host_targets, Some(true));
        state.bind_targets(&list, false);
        assert_eq!(state.depth, 0);
        assert_eq!(state.host_targets, Some(false));
        state.bind_targets(&list, true);
        assert_eq!(state.depth, depth.0 as usize);

        let continued = host.new_list().expect("continued list");
        crate::descriptor::create_rtv(std::ptr::null(), large.0, std::ptr::null(), small_handle);
        crate::descriptor::create_dsv(
            std::ptr::null(),
            replacement_depth.0,
            std::ptr::null(),
            depth_handle,
        );
        state.replay(&continued);
        assert_eq!(state.host_targets, None);
        assert_eq!(state.depth, 0);
        state.bind_targets(&continued, true);
        assert_eq!(state.targets, [small.0 as usize]);
        assert_eq!(state.depth, depth.0 as usize);
        state.set_targets(&continued, &[large_handle], depth_handle);
        state.bind_targets(&continued, false);
        assert_eq!(state.depth, 0);
        list.close();
        continued.close();
        host.execute(&[&list, &continued]);
        unsafe { host.device.GetDeviceRemovedReason() }.expect("target binding removed the device");
        assert!(
            host.log_messages().is_none(),
            "target binding validation error"
        );
    }
}
