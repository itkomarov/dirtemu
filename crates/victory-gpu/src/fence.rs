//! Fences: a 64-bit value and the events waiting for it. Nothing runs on a
//! GPU yet, so only signals (the title's, later our queues') move one.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
#[cfg(windows)]
use std::sync::Arc;
use std::sync::Mutex;
#[cfg(windows)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::{self, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    waiters: Mutex<Vec<(u64, Completion)>>,
}

enum Completion {
    Event(usize),
    #[cfg(windows)]
    Blocking(Arc<AtomicBool>),
}

#[cfg(windows)]
fn wait_for_value(fence: &Fence, value: u64, reached: &AtomicBool) -> HRESULT {
    use windows_sys::Win32::Foundation::{ERROR_TIMEOUT, GetLastError};
    use windows_sys::Win32::System::Threading::WaitOnAddress;

    loop {
        if reached.load(Ordering::Acquire) || fence.value.load(Ordering::Acquire) >= value {
            return S_OK;
        }
        if crate::host::late() {
            crate::host::settle_for_fence();
        }
        crate::queue::release_held(false);
        let current = fence.value.load(Ordering::Acquire);
        if reached.load(Ordering::Acquire) || current >= value {
            return S_OK;
        }
        let waited = unsafe {
            WaitOnAddress(
                (&raw const fence.value).cast(),
                (&raw const current).cast(),
                std::mem::size_of::<u64>(),
                100,
            )
        };
        if waited == 0 {
            let error = unsafe { GetLastError() };
            if error != ERROR_TIMEOUT {
                tracing::warn!(value, error, "blocking fence wait failed");
                return if error == 0 {
                    0x8000_4005u32 as HRESULT
                } else {
                    (0x8007_0000 | (error & 0xffff)) as HRESULT
                };
            }
        }
    }
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        #[cfg(windows)]
        {
            let reached = Arc::new(AtomicBool::new(false));
            waiters.push((value, Completion::Blocking(Arc::clone(&reached))));
            drop(waiters);
            let result = wait_for_value(fence, value, &reached);
            let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
            waiters.retain(|(_, completion)| {
                !matches!(completion, Completion::Blocking(other) if Arc::ptr_eq(other, &reached))
            });
            return result;
        }
        #[cfg(not(windows))]
        return object::E_NOTIMPL;
    }
    waiters.push((value, Completion::Event(event as usize)));
    S_OK
}
/// Moves the fence to `value` and sets the events that waited for it.
pub(crate) fn set(this: This, value: u64) {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|(wanted, completion)| {
        let reached = *wanted <= value;
        if reached {
            match completion {
                Completion::Event(event) => set_event(*event),
                #[cfg(windows)]
                Completion::Blocking(reached) => reached.store(true, Ordering::Release),
            }
        }
        !reached
    });
    drop(waiters);
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Threading::WakeByAddressAll((&raw const fence.value).cast());
    }
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    set(this, value);
    // A queue may have waited for it: what it held runs now.
    #[cfg(windows)]
    crate::queue::release_held(false);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, std::ptr::null_mut()), S_OK);
        // The title reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }
    #[cfg(windows)]
    struct BlockingWait {
        fence: usize,
        done: std::sync::mpsc::Receiver<HRESULT>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    #[cfg(windows)]
    impl BlockingWait {
        fn start(fence: This, value: u64) -> Self {
            let (sent, done) = std::sync::mpsc::channel();
            object::add_ref(fence);
            let at = fence as usize;
            let thread = std::thread::spawn(move || {
                let result = set_event_on_completion(at as This, value, std::ptr::null_mut());
                release(at as This);
                let _ = sent.send(result);
            });
            Self {
                fence: at,
                done,
                thread: Some(thread),
            }
        }

        fn registered(&self) {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                let registered = fence(self.fence as This)
                    .waiters
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .any(|(_, completion)| matches!(completion, Completion::Blocking(_)));
                if registered {
                    return;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "NULL wait was not registered"
                );
                assert!(
                    matches!(
                        self.done.try_recv(),
                        Err(std::sync::mpsc::TryRecvError::Empty)
                    ),
                    "NULL wait returned before its target"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        fn pending(&self) {
            assert!(
                matches!(
                    self.done.recv_timeout(std::time::Duration::from_millis(30)),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                ),
                "NULL wait returned before its target"
            );
        }

        fn finish(mut self) -> HRESULT {
            let result = self.done.recv_timeout(std::time::Duration::from_secs(2));
            if result.is_err() {
                set(self.fence as This, u64::MAX);
            }
            let result = result.expect("NULL wait did not complete after the signal");
            self.thread.take().unwrap().join().unwrap();
            result
        }
    }

    #[cfg(windows)]
    impl Drop for BlockingWait {
        fn drop(&mut self) {
            if let Some(thread) = self.thread.take() {
                set(self.fence as This, u64::MAX);
                if self
                    .done
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .is_ok()
                {
                    let _ = thread.join();
                }
            }
        }
    }

    #[cfg(windows)]
    fn new_fence(initial: u64) -> This {
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), initial, 0, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        f
    }

    #[cfg(windows)]
    #[test]
    fn null_wait_blocks_until_cpu_signal_reaches_target() {
        let f = new_fence(1);
        let waiting = BlockingWait::start(f, 5);
        waiting.registered();
        waiting.pending();
        assert_eq!(signal(f, 4), S_OK);
        waiting.pending();
        assert_eq!(signal(f, 5), S_OK);
        assert_eq!(waiting.finish(), S_OK);
        assert!(fence(f).waiters.lock().unwrap().is_empty());
        assert_eq!(release(f), 0);
    }

    #[cfg(windows)]
    #[test]
    fn null_wait_completes_from_queue_written_value() {
        let f = new_fence(0);
        let waiting = BlockingWait::start(f, 9);
        waiting.registered();
        waiting.pending();
        crate::queue::Written::Value(9).write(gpu_address(f) as *mut u8, 8);
        assert_eq!(waiting.finish(), S_OK);
        assert_eq!(completed(f), 9);
        assert!(fence(f).waiters.lock().unwrap().is_empty());
        assert_eq!(release(f), 0);
    }

    #[cfg(windows)]
    #[test]
    fn null_wait_observes_raw_value_without_a_wakeup() {
        let f = new_fence(0);
        let waiting = BlockingWait::start(f, 9);
        waiting.registered();
        waiting.pending();
        fence(f).value.store(9, Ordering::Release);
        assert_eq!(waiting.finish(), S_OK);
        assert_eq!(release(f), 0);
    }

    #[cfg(windows)]
    #[test]
    fn null_wait_completion_survives_fence_rewind() {
        let f = new_fence(0);
        let waiting = BlockingWait::start(f, 5);
        waiting.registered();
        waiting.pending();
        let reached = {
            let waiters = fence(f).waiters.lock().unwrap();
            match &waiters[0].1 {
                Completion::Blocking(reached) => Arc::clone(reached),
                Completion::Event(_) => unreachable!(),
            }
        };
        set(f, 5);
        set(f, 0);
        assert!(reached.load(Ordering::Acquire));
        assert_eq!(wait_for_value(fence(f), 5, &reached), S_OK);
        assert_eq!(waiting.finish(), S_OK);
        assert_eq!(completed(f), 0);
        assert_eq!(release(f), 0);
    }

    #[cfg(windows)]
    #[test]
    fn null_wait_wakes_all_waiters_at_their_own_targets() {
        let f = new_fence(0);
        let first = BlockingWait::start(f, 3);
        first.registered();
        let second = BlockingWait::start(f, 7);
        second.registered();
        first.pending();
        second.pending();
        set(f, 3);
        assert_eq!(first.finish(), S_OK);
        second.pending();
        set(f, 7);
        assert_eq!(second.finish(), S_OK);
        assert_eq!(release(f), 0);
    }
}
