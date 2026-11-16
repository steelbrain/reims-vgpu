//! What the decoder and encoder sessions share: a session's handle, which the host keeps by its
//! UUID and hands each of the session's messages to, the session's thread's queue, the link the
//! host's messages go on, and the host's requests, which wait for the guest's reply.
//!
//! The guest's VideoToolbox sends a session's messages to the host, which answers each on the
//! session's thread, in order. The host's own requests (flag `HOST_EXPECTS_REPLY`) wait for the
//! guest's reply (flag `REPLY_TO_HOST`), which goes straight to the request that waits for it
//! ([`Session::deliver`]), not through the session's queue.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use crate::core::fourcc::FourCc;
use crate::core::runtime::{self, Answerer, Runtime};
use crate::core::sync::Mutex;
use crate::paravirt::message::{GuestUuid, MessageFlags, MessageHeader, MessageWriter};
use crate::paravirt::status::{self, OsStatus};

/// A guest surface a message lent the host, as the device resolved it: held by the host until it
/// lets go of it with a `rels` of [`LentSurface::id`].
pub trait LentSurface: Send + Sync + std::fmt::Debug {
    /// The name the guest's kext gives the surface, which a `rels` names it by.
    fn id(&self) -> u64;
    /// The device's own type, for its backend to read.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The way the host's messages reach the guest's VideoToolbox, from any thread.
pub trait Link: Send + Sync {
    /// Sends `message`, a `vtpv` message or a `rels` box; false where it was not sent.
    fn send(&self, message: &[u8]) -> bool;
}

/// The `rels` box that tells the guest the host holds these surfaces no longer: a big-endian size
/// of 8 plus 8 an ID, the type, then each ID big-endian. It travels alone, not inside a `vtpv`
/// message.
pub fn surface_release(ids: &[u64]) -> Vec<u8> {
    let size = u32::try_from(8 + 8 * ids.len()).unwrap_or(u32::MAX);
    let mut bytes = Vec::with_capacity(8 + 8 * ids.len());
    bytes.extend_from_slice(&size.to_be_bytes());
    bytes.extend_from_slice(b"rels");
    for id in ids {
        bytes.extend_from_slice(&id.to_be_bytes());
    }
    bytes
}

/// The guest's reply to a host request, with the surfaces it lent.
pub(crate) struct GuestReply {
    pub(crate) bytes: Vec<u8>,
    pub(crate) surfaces: Vec<Arc<dyn LentSurface>>,
}

/// What a session's handle and its thread share.
pub(crate) struct Shared {
    pub(crate) uuid: GuestUuid,
    /// The last link a message came with.
    link: Mutex<Option<Arc<dyn Link>>>,
    /// The host's requests waiting for the guest's reply, by reply identifier.
    waiting: Mutex<HashMap<u64, Answerer<GuestReply>>>,
    /// The next host request's reply identifier.
    next_request: AtomicU64,
}

impl Shared {
    /// Sends `message` on the last link; false where there is none or it was not sent.
    pub(crate) fn send(&self, message: &[u8]) -> bool {
        let link = self.link.lock().clone();
        link.is_some_and(|link| link.send(message))
    }

    /// Lets go of `surfaces`, where there are any: a `rels` of their IDs.
    pub(crate) fn let_go(&self, surfaces: &[Arc<dyn LentSurface>]) {
        if surfaces.is_empty() {
            return;
        }
        let ids: Vec<u64> = surfaces.iter().map(|surface| surface.id()).collect();
        if !self.send(&surface_release(&ids)) {
            log::warn!(
                "hardware video: a release of {} lent surfaces was not sent",
                ids.len()
            );
        }
    }

    /// A host request of type `kind`, flagged `HOST_EXPECTS_REPLY` and filled by `fill`, and the
    /// reply identifier of its own it carries.
    pub(crate) fn request_message(
        &self,
        kind: FourCc,
        fill: impl FnOnce(&mut MessageWriter),
    ) -> (u64, MessageWriter) {
        // Past 2^64 requests the count goes round, past 0.
        let id = self.next_request.fetch_add(1, Ordering::Relaxed).max(1);
        let mut message =
            MessageWriter::new(kind, MessageFlags::HOST_EXPECTS_REPLY, Some(self.uuid), id);
        fill(&mut message);
        (id, message)
    }

    /// Sends `before` as they are, then `last`, the request of reply identifier `id`, and waits
    /// `wait` on `runtime` for the guest's reply to it; `kVTParameterErr` where a message was not
    /// sent or the reply did not come in time.
    pub(crate) fn request(
        &self,
        runtime: &dyn Runtime,
        id: u64,
        before: &[Vec<u8>],
        last: &[u8],
        wait: Duration,
    ) -> Result<GuestReply, OsStatus> {
        let kind = crate::paravirt::message::read_header(last, 0)
            .map(|header| header.message_type)
            .unwrap_or(FourCc(0));
        let (answerer, answer) = runtime::answer();
        self.waiting.lock().insert(id, answerer);
        let sent = before.iter().all(|message| self.send(message)) && self.send(last);
        if !sent {
            self.waiting.lock().remove(&id);
            return Err(status::PARAMETER);
        }
        let deadline = runtime.deadline_in(wait);
        answer.wait_until(runtime, deadline).map_err(|unanswered| {
            self.waiting.lock().remove(&id);
            log::warn!("hardware video: no answer to {kind} in {wait:?}: {unanswered:?}");
            status::PARAMETER
        })
    }
}

/// A message for a session's thread.
pub(crate) struct Job {
    pub(crate) bytes: Vec<u8>,
    pub(crate) surfaces: Vec<Arc<dyn LentSurface>>,
}

/// A session's handle, which the host keeps by its UUID. Dropping it ends the session's thread
/// once its current message is done.
pub(crate) struct Session {
    queue: Sender<Job>,
    shared: Arc<Shared>,
}

impl Session {
    /// A session for `uuid`, whose thread, named `thread`, runs `worker` on `runtime` with what
    /// the handle shares and its queue.
    pub(crate) fn start(
        uuid: GuestUuid,
        thread: &'static str,
        runtime: &dyn Runtime,
        link: Option<Arc<dyn Link>>,
        worker: impl FnOnce(Arc<Shared>, Receiver<Job>) + Send + 'static,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            uuid,
            link: Mutex::new(link),
            waiting: Mutex::new(HashMap::new()),
            next_request: AtomicU64::new(1),
        });
        let (queue, jobs) = channel();
        let theirs = Arc::clone(&shared);
        runtime.spawn(thread, Box::new(move || worker(theirs, jobs)))?;
        Ok(Self { queue, shared })
    }

    /// Takes the guest's message `bytes` for the session, with the surfaces it lent and the link
    /// it came on: a reply to a host request goes to the request, any other message to the
    /// session's queue. noErr, or [`status::SESSION_NOT_FOUND`] where the session's thread has
    /// ended.
    pub(crate) fn deliver(
        &self,
        header: &MessageHeader,
        bytes: &[u8],
        surfaces: Vec<Arc<dyn LentSurface>>,
        link: Option<Arc<dyn Link>>,
    ) -> OsStatus {
        if let Some(link) = link {
            *self.shared.link.lock() = Some(link);
        }
        if header.flags.contains(MessageFlags::REPLY_TO_HOST) {
            let waiting = self.shared.waiting.lock().remove(&header.reply_id);
            match waiting {
                Some(answerer) => answerer.give(GuestReply {
                    bytes: bytes.to_vec(),
                    surfaces,
                }),
                // A reply past its request's wait: what it lent goes back.
                None => self.shared.let_go(&surfaces),
            }
            return status::NO_ERROR;
        }
        let job = Job {
            bytes: bytes.to_vec(),
            surfaces,
        };
        match self.queue.send(job) {
            Ok(()) => status::NO_ERROR,
            Err(std::sync::mpsc::SendError(job)) => {
                self.shared.let_go(&job.surfaces);
                status::SESSION_NOT_FOUND
            }
        }
    }
}
