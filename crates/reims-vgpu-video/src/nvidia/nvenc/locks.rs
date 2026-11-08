//! A session's lock thread, which makes the session's waiting
//! `nvEncLockBitstream` calls, so that the session's own thread waits for
//! each with a bound (`crate::nvidia::wait`).
//!
//! NVENC's lock has no bounded form here. A session runs in synchronous
//! mode, the only one on Linux (`enableEncodeAsync` and its completion
//! events are Windows's), and there a lock with `doNotWait` waits for its
//! picture as a waiting lock does, and gives an IDR picture wrongly.
//! Measured on an RTX 5080 Laptop GPU, driver 615.71.09, 2026-10-25, 4K
//! HEVC Main with no B pictures, each lock made as soon as
//! `nvEncEncodePicture` returned, which took 25 to 78 µs
//! (`encoder::tests::what_each_lock_returns`, and a program of our own that
//! compared each picture with the same picture of a second session;
//! docs/video/bugs.md, U19):
//! - a lock with `doNotWait` returned once its picture was encoded, 6.0 to
//!   16.4 ms after the call, as a waiting lock did; it never answered
//!   `NV_ENC_ERR_LOCK_BUSY`;
//! - without split-frame encoding (P4 at high quality at 4K, as here), it
//!   answered an IDR picture's `bitstreamSizeInBytes` 200 to 400 times the
//!   picture's, with the right bytes: 391,140,259 to 398,125,977, where
//!   either lock made 40 ms later answered 1,031,535 to 1,032,383; for the
//!   other pictures its sizes were those of a later lock's;
//! - with split-frame encoding (P4 at low latency at 4K, which turns it on,
//!   or two strips forced), it answered an IDR picture's size right and
//!   about half its bytes wrong: a second VPS, SPS and PPS and 165 zero
//!   bytes before the first strip, which ffmpeg decoded, with no error, to
//!   another picture;
//! - a waiting lock made at once gave the right size and bytes, and either
//!   lock made 40 ms later answered in 3 to 5 µs;
//! - a second lock of a buffer already unlocked, with no picture submitted
//!   into it since, never returned, with `doNotWait` or without it: one
//!   spun a CPU core for 14 minutes. So a session locks a buffer only while
//!   a picture submitted into it waits for its lock (`Submitted`).
//!
//! So the waiting lock is made here, on a thread of the session's own
//! (`crate::nvidia::wait`'s `Calls`), and the session's thread waits for its answer
//! until the deadline. A lock's round trip costs the session's thread two
//! wakes of a thread.
//!
//! **Past the deadline** the thread stays in the lock, and the session is
//! stalled (`Session`): its thread makes no more NVENC calls. When the
//! session drops, it hands the thread its handles, buffers and
//! configuration ([`Finishing`]). Once the lock returns, the thread unlocks
//! the buffer, ends the stream and locks each other buffer still submitted,
//! as an encoder drains its session before it is destroyed (`encoder`'s
//! `Drop`), then destroys the session and ends. A lock that never returns
//! keeps the thread, the session and its buffers until the process ends;
//! `crate::nvidia::wait`'s list keeps a record of it for as long.

use std::ffi::c_void;
use std::sync::Arc;

use crate::core::runtime::{Answer, MonoTime, Unanswered};

use super::{Session, ffi};
use crate::nvidia::cuda::Gpu;
use crate::nvidia::dl::Zeroed;
use crate::nvidia::error::Error;
use crate::nvidia::wait::{self, Calls, Unreturned};

/// The thread's name (ours).
const THREAD_NAME: &str = "reims-vgpu-video.nvenc-locks";

/// `nvEncLockBitstream`'s entry point.
pub(super) type LockCall =
    unsafe extern "C" fn(ffi::Encoder, *mut ffi::NV_ENC_LOCK_BITSTREAM) -> ffi::NVENCSTATUS;

/// What a waiting lock answered: its status, and what NVENC filled in.
pub(super) struct Lock {
    pub(super) status: ffi::NVENCSTATUS,
    pub(super) locked: ffi::NV_ENC_LOCK_BITSTREAM,
}

// SAFETY: the pointers are NVENC's, into the buffer the lock locked, which
// one thread reads at a time: the session's, once it has the answer, or
// this one's while finishing.
unsafe impl Send for Lock {}

/// The answer a lock order gives.
pub(super) type Locking = Answer<Result<Lock, Error>>;

/// A waiting lock of `output`, a buffer of the live session `encoder` that
/// a submitted picture fills, with `gpu`'s context pushed.
pub(super) fn lock_now(
    call: LockCall,
    encoder: ffi::Encoder,
    gpu: &Gpu,
    output: *mut c_void,
) -> Result<Lock, Error> {
    let _current = gpu.push()?;
    let mut locked = ffi::NV_ENC_LOCK_BITSTREAM::zeroed();
    locked.version = ffi::NV_ENC_LOCK_BITSTREAM_VER;
    locked.outputBitstream = output;
    // SAFETY: a live session and one of its output buffers, which a
    // submitted picture fills; the lock waits for it.
    let status = unsafe { call(encoder, &raw mut locked) };
    Ok(Lock { status, locked })
}

/// A session's lock thread (`crate::nvidia::wait`'s [`Calls`]), started with the
/// session's first lock; it ends as this drops, once it has made what it
/// was ordered: a lock it answers, or the session it finishes.
pub(super) struct Locks {
    encoder: ffi::Encoder,
    gpu: Arc<Gpu>,
    calls: Calls,
}

/// A lock the thread makes.
struct LockOrder {
    call: LockCall,
    encoder: ffi::Encoder,
    gpu: Arc<Gpu>,
    output: *mut c_void,
}

// SAFETY: the session's handle and its buffer's are the driver's, called one
// call at a time: by the thread while the session's thread waits for its
// answer, or once the session has stalled, by the thread alone (module
// docs).
unsafe impl Send for LockOrder {}

impl LockOrder {
    fn make(self) -> Result<Lock, Error> {
        lock_now(self.call, self.encoder, &self.gpu, self.output)
    }
}

impl Locks {
    /// Starts the thread, for the session `encoder` on `gpu`'s context:
    /// `None` where no thread starts, and the session locks on its own
    /// thread, with no bound.
    pub(super) fn start(encoder: ffi::Encoder, gpu: &Arc<Gpu>) -> Option<Self> {
        Some(Self {
            encoder,
            gpu: Arc::clone(gpu),
            calls: Calls::start(THREAD_NAME)?,
        })
    }

    /// Orders a waiting lock of `output` with `call`: its answer.
    pub(super) fn lock(&self, call: LockCall, output: *mut c_void) -> Locking {
        let order = LockOrder {
            call,
            encoder: self.encoder,
            gpu: Arc::clone(&self.gpu),
            output,
        };
        self.calls.call(move || order.make())
    }

    /// Gives the thread the stalled session to finish once its lock
    /// returns (module docs): the record of it that `crate::nvidia::wait`'s list
    /// keeps until the thread has destroyed it.
    pub(super) fn finish(self, finishing: Finishing) -> Unreturned<()> {
        let finished = self.calls.call(move || finishing.finish());
        Unreturned::new(finished, Vec::new())
    }
}

/// A stalled session, which its lock thread finishes and destroys
/// (module docs).
pub(super) struct Finishing {
    /// The session, with what it may still write: not stalled, and with no
    /// thread of its own, so that its locks wait on the thread finishing it.
    pub(super) session: Session,
    /// The buffer whose lock passed its deadline, and that lock's answer.
    pub(super) output: *mut c_void,
    pub(super) locking: Locking,
}

// SAFETY: the session's handles, called by the thread finishing it alone.
unsafe impl Send for Finishing {}

impl Finishing {
    fn finish(self) {
        let Self {
            session,
            output,
            locking,
        } = self;
        // The thread answered the lock before it took this order, or the
        // order took the place of a lock it had not taken, which answers
        // that none is to come: then the buffer is locked with the rest.
        let made = match locking.wait_until(&**wait::runtime(), MonoTime::END) {
            Ok(Ok(lock)) => {
                if lock.status == ffi::NV_ENC_SUCCESS {
                    session.unlock_bitstream(output);
                }
                true
            }
            Ok(Err(_)) | Err(Unanswered::TimedOut | Unanswered::Gone) => false,
        };
        if made {
            session.submitted.borrow_mut().locked(output);
        }
        session.drain();
        drop(session);
    }
}
