//! The generation of the runtime's device that a user of its raw Vulkan handles makes its own
//! objects on ([`Generation`]): the plane copier's (`crate::os::PlaneCopier`), the toucher that
//! keeps the GPU awake (`crate::awake`), and a window's swapchain (`crate::swapchain`).
//!
//! - **Why.** The runtime's device is a line of generations, one `VkDevice` each. Once its
//!   `VkDevice` is lost, the runtime makes a new one, and every later call goes there
//!   (metal2vulkan's `device/line.rs`; `rt::Device::recover_for_test` runs the same recovery with
//!   no fault). A root handle's `rt::Device::raw` answers the generation that serves at the call,
//!   and lets go of the one it answered before; `rt::CommandQueue::with_vk_queue` hands out the
//!   `VkQueue` of the queue's place on the generation that serves. So an object made with one
//!   generation's handles was used, after a recovery, through the next one's or submitted to its
//!   queue: objects of two devices in one call. Measured on a Radeon 780M (RADV, Mesa 26.2.6), the
//!   plane copier's copy after the runtime made its `VkDevice` again died in the driver, on a read
//!   at address 0x50.
//! - **Its handles.** [`Generation::serving`] takes the generation that serves, as the device of
//!   an object made on it (a Metal fence, which makes no Vulkan object): a generation handle,
//!   whose `rt::Device::raw` answers that generation's `VkDevice` at every call, and which keeps
//!   it alive while the user holds it. So the user makes its objects with those handles, and
//!   destroys them with the same handles after a recovery too, before it lets the generation go.
//! - **Its queue.** With it the user takes the `VkQueue` its runtime queue submits to there, and
//!   makes each submission inside `with_vk_queue` only where the queue handed in is that one
//!   ([`Generation::with_queue`]). Another is the queue's place on a newer generation: the kept
//!   `VkDevice`'s queues live as long as it does, so a newer device's never has one of their
//!   handles. The user then makes its objects again on the generation that serves.
//! - **The objects it meets.** A runtime object's device is the generation it was made on: a
//!   runtime buffer the user's commands write is checked to be of the user's generation
//!   ([`Generation::is_of`]) before its handle is recorded.

use ash::vk;
use metal2vulkan_runtime as rt;

/// How many times [`Generation::serving`] takes a generation and its queue before it gives up:
/// each turn but the last was a recovery between the two.
const TAKES: u32 = 4;

/// One generation of a runtime device, kept alive, and the `VkQueue` a runtime queue of the
/// device submits to on it (the module docs).
pub(crate) struct Generation {
    /// A generation handle: its raw handles are this generation's.
    device: rt::Device,
    /// Its `VkDevice`, by which generations are told apart while both live.
    handle: vk::Device,
    /// The `VkQueue` the queue submits to on it.
    queue: vk::Queue,
    /// That queue's family.
    family: u32,
}

impl std::fmt::Debug for Generation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generation")
            .field("handle", &self.handle)
            .field("queue", &self.queue)
            .field("family", &self.family)
            .finish_non_exhaustive()
    }
}

/// The generation of `device`'s line that serves now: the device of a fence made on it.
fn generation_serving(device: &rt::Device) -> rt::Device {
    device.new_fence().device().clone()
}

/// The `VkDevice` of `generation`, a generation handle (an object's device, or one
/// [`generation_serving`] took): its own, which a call on a root handle would not be.
fn handle_of(generation: &rt::Device) -> vk::Device {
    // SAFETY: only the handle is read, while the generation lives.
    unsafe { generation.raw() }.device.handle()
}

impl Generation {
    /// The generation of `queue`'s device that serves, with the `VkQueue` `queue` submits to there
    /// and its family; `Err` where the device recovered each time the two were taken
    /// ([`TAKES`]). The generation is taken before the queue and again after it: where both are
    /// the same, the queue's place is on it.
    pub(crate) fn serving(queue: &rt::CommandQueue) -> Result<Generation, String> {
        for _ in 0..TAKES {
            let device = generation_serving(queue.device());
            // SAFETY: the closure keeps no handle and calls nothing.
            let (vk_queue, family) = unsafe { queue.with_vk_queue(|_, q, family| (q, family)) };
            let handle = handle_of(&device);
            if handle_of(&generation_serving(queue.device())) == handle {
                return Ok(Generation {
                    device,
                    handle,
                    queue: vk_queue,
                    family,
                });
            }
        }
        Err(format!(
            "the device recovered each of the {TAKES} times its queue was taken"
        ))
    }

    /// The generation handle: its calls go to this generation while it serves.
    pub(crate) fn device(&self) -> &rt::Device {
        &self.device
    }

    /// This generation's raw handles.
    ///
    /// # Safety
    ///
    /// `rt::Device::raw`'s: the handles are used while `self` lives, and the queues the runtime
    /// submits to only through [`Generation::with_queue`].
    pub(crate) unsafe fn raw(&self) -> rt::DeviceHandles<'_> {
        // SAFETY: the caller's.
        unsafe { self.device.raw() }
    }

    /// The family of the queue's `VkQueue` here.
    pub(crate) fn family(&self) -> u32 {
        self.family
    }

    /// Whether this generation still serves `queue`'s device: false once a recovery made a new
    /// `VkDevice`, and the user's objects are to be made again ([`Generation::serving`]).
    pub(crate) fn serves(&self, queue: &rt::CommandQueue) -> bool {
        handle_of(&generation_serving(queue.device())) == self.handle
    }

    /// Whether `device`, the device of a runtime object (a generation handle), is this
    /// generation: the object's Vulkan handles are of this generation's `VkDevice`.
    pub(crate) fn is_of(&self, device: &rt::Device) -> bool {
        handle_of(device) == self.handle
    }

    /// `submit` with this generation's `VkQueue`, inside `queue`'s `with_vk_queue`, under the
    /// runtime's lock of that queue: `None`, with nothing called, where `queue` submits to
    /// another generation's queue now (a recovery moved it there).
    ///
    /// # Safety
    ///
    /// `rt::CommandQueue::with_vk_queue`'s: `submit` keeps no handle, makes no call into the
    /// runtime and waits for nothing a later submission of the runtime's gives. `queue` is the one
    /// this generation was taken with.
    pub(crate) unsafe fn with_queue<R>(
        &self,
        queue: &rt::CommandQueue,
        submit: impl FnOnce(vk::Queue) -> R,
    ) -> Option<R> {
        // SAFETY: the caller's.
        unsafe { queue.with_vk_queue(|_, q, _| (q == self.queue).then(|| submit(q))) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device of its own for each test that recovers it, as a recovery moves every user of the
    /// device: the process's test device stays as it was.
    fn own_device() -> rt::Device {
        rt::Device::system_default().expect("a runtime device")
    }

    /// A generation serves until the device recovers; then its queue is another generation's, so
    /// nothing is submitted with it, and a generation taken again serves, of another `VkDevice`
    /// and with another queue. An object made before the recovery is of the old one, and one made
    /// after of the new.
    #[test]
    fn a_generation_serves_until_its_device_recovers() {
        let device = own_device();
        let queue = device.new_command_queue().expect("a queue");
        let first = Generation::serving(&queue).expect("a generation");
        assert!(first.serves(&queue));
        // SAFETY: the closure keeps the handle only to return it, compared and never used.
        let submitted = unsafe { first.with_queue(&queue, |q| q) };
        assert_eq!(submitted, Some(first.queue));
        let before = device.new_fence();
        assert!(first.is_of(before.device()));

        device.recover_for_test().expect("the recovery");
        assert!(!first.serves(&queue));
        // SAFETY: as above.
        assert_eq!(unsafe { first.with_queue(&queue, |q| q) }, None);
        assert!(
            first.is_of(before.device()),
            "the old object is the old generation's"
        );

        let second = Generation::serving(&queue).expect("a generation");
        assert!(second.serves(&queue));
        assert_ne!(second.handle, first.handle);
        assert_ne!(second.queue, first.queue);
        // SAFETY: as above.
        assert_eq!(
            unsafe { second.with_queue(&queue, |q| q) },
            Some(second.queue)
        );
        let after = device.new_fence();
        assert!(second.is_of(after.device()));
        assert!(!second.is_of(before.device()));
        // The old generation's handles still answer its own `VkDevice` while it is held.
        // SAFETY: only the handle is read, while `first` lives.
        assert_eq!(unsafe { first.raw() }.device.handle(), first.handle);
    }
}
