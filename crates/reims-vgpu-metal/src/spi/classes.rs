//! Metal's private classes the platform names (the buffer descriptor, the address-range array,
//! the ICB argument classes) with typed declarations of the methods it sends them, the
//! deserialization context that answers a pipeline descriptor's function references.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;

use objc2::exception::Exception;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{AllocAnyThread, DefinedClass, define_class, extern_class, extern_methods, msg_send};
use objc2_foundation::{NSException, NSString};
use objc2_metal::{MTLFunction, MTLSize};
use reims_vgpu_core::platform::types::ResourceAddressRange;
use reims_vgpu_core::raise::Raise;
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::rt::Hold;

use crate::gpu::resource::MetalFunction;
use crate::msg::encode::Enc;
use crate::spi::methods::Deallocator;

extern_class!(
    /// The private `MTLBufferDescriptor` (the images import `_OBJC_CLASS_$_MTLBufferDescriptor`).
    #[unsafe(super(NSObject))]
    #[name = "MTLBufferDescriptor"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct MTLBufferDescriptor;
);

#[allow(non_snake_case, missing_docs, reason = "each method is its selector")]
impl MTLBufferDescriptor {
    extern_methods!(
        /// `+new`: `+1`.
        #[unsafe(method(new))]
        pub fn new() -> Retained<Self>;

        /// The buffer's bytes, as they are (no copy with `noCopy`).
        ///
        /// # Safety
        /// `contents` stays valid for the buffer made from the descriptor.
        #[unsafe(method(setContents:))]
        pub unsafe fn setContents(&self, contents: *mut c_void);

        /// Metal copies the block and calls it once, when the buffer it made is released.
        #[unsafe(method(setDeallocator:))]
        pub fn setDeallocator(&self, deallocator: Option<&Deallocator>);

        #[unsafe(method(setLength:))]
        pub fn setLength(&self, length: u64);

        #[unsafe(method(setNoCopy:))]
        pub fn setNoCopy(&self, no_copy: bool);

        #[unsafe(method(setPinnedGPUAddress:))]
        pub fn setPinnedGPUAddress(&self, address: u64);

        #[unsafe(method(setResourceOptions:))]
        pub fn setResourceOptions(&self, options: u64);

        #[unsafe(method(setStorageMode:))]
        pub fn setStorageMode(&self, mode: u64);

        #[unsafe(method(resourceOptions))]
        pub fn resourceOptions(&self) -> u64;
    );
}

extern_class!(
    /// The private `MTLResourceAddressRangeArray`.
    #[unsafe(super(NSObject))]
    #[name = "MTLResourceAddressRangeArray"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct MTLResourceAddressRangeArray;
);

#[allow(non_snake_case, missing_docs, reason = "each method is its selector")]
impl MTLResourceAddressRangeArray {
    extern_methods!(
        /// The array copies the ranges (`platform`'s test `new_address_range_array`).
        ///
        /// # Safety
        /// `ranges` is `count` readable ranges.
        #[unsafe(method(initWithRanges:count:))]
        pub unsafe fn initWithRanges_count(
            this: Allocated<Self>,
            ranges: *const Enc<ResourceAddressRange>,
            count: u64,
        ) -> Option<Retained<Self>>;

        #[unsafe(method(count))]
        pub fn count(&self) -> u64;

        /// The array's own copy of the ranges.
        #[unsafe(method(ranges))]
        pub fn ranges(&self) -> *const Enc<ResourceAddressRange>;
    );
}

macro_rules! private_classes {
    ($($rust:ident = $name:literal { $($(#[$m:meta])* fn $method:ident -> $ret:ty;)* })*) => {
        $(
            extern_class!(
                #[doc = concat!("Metal's private `", $name, "`.")]
                #[unsafe(super(NSObject))]
                #[name = $name]
                #[derive(Debug, PartialEq, Eq, Hash)]
                pub struct $rust;
            );

            #[allow(non_snake_case, missing_docs, reason = "each getter is its selector")]
            impl $rust {
                extern_methods!(
                    $(
                        $(#[$m])*
                        #[unsafe(method($method))]
                        pub fn $method(&self) -> $ret;
                    )*
                );
            }
        )*
    };
}

private_classes! {
    IndirectDrawArguments = "_MTLIndirectDrawArguments" {
        fn baseInstance -> u64;
        fn instanceCount -> u64;
        fn primitiveType -> u64;
        fn vertexCount -> u64;
        fn vertexStart -> u64;
    }
    IndirectDrawIndexedArguments = "_MTLIndirectDrawIndexedArguments" {
        fn baseInstance -> u64;
        fn baseVertex -> u64;
        fn indexBufferGPUVirtualAddress -> u64;
        fn indexCount -> u64;
        fn indexType -> u64;
        fn instanceCount -> u64;
        fn primitiveType -> u64;
    }
    IndirectDrawPatchesArguments = "_MTLIndirectDrawPatchesArguments" {
        fn baseInstance -> u64;
        fn instanceCount -> u64;
        fn numberOfPatchControlPoints -> u64;
        fn patchCount -> u64;
        fn patchIndexBufferGPUVirtualAddress -> u64;
        fn patchStart -> u64;
    }
    IndirectDrawIndexedPatchesArguments = "_MTLIndirectDrawIndexedPatchesArguments" {
        fn baseInstance -> u64;
        fn controlPointIndexBufferGPUVirtualAddress -> u64;
        fn instanceCount -> u64;
        fn numberOfPatchControlPoints -> u64;
        fn patchCount -> u64;
        fn patchIndexBufferGPUVirtualAddress -> u64;
        fn patchStart -> u64;
    }
    IndirectTessellationFactorArguments = "_MTLIndirectTessellationFactorArguments" {
        fn GPUVirtualAddress -> u64;
        fn instanceStride -> u64;
        fn scale -> f32;
    }
    IndirectDispatchThreadsArguments = "_MTLIndirectDispatchThreadsArguments" {
        fn threadsPerGrid -> MTLSize;
        fn threadsPerThreadgroup -> MTLSize;
    }
    IndirectDispatchThreadgroupsArguments = "_MTLIndirectDispatchThreadgroupsArguments" {
        fn threadgroupsPerGrid -> MTLSize;
        fn threadsPerThreadgroup -> MTLSize;
    }
    ImageBlockArguments = "_MTLImageBlockArguments" {
        fn height -> u64;
        fn width -> u64;
    }
}

/// The protocol, in its own module: objc2's macro drops the trait's `# Safety` docs.
#[allow(clippy::missing_safety_doc)]
mod protocol {
    use objc2::extern_protocol;
    use objc2::runtime::NSObjectProtocol;

    extern_protocol!(
        /// `@protocol MTLDeserializationContext <NSObject>`: what a deserialized pipeline
        /// descriptor asks for its functions.
        ///
        /// # Safety
        /// An implementor answers `functionWithFunctionRef:` with nil or an `id<MTLFunction>`.
        #[name = "MTLDeserializationContext"]
        pub unsafe trait MTLDeserializationContext: NSObjectProtocol {}
    );
}
pub use protocol::MTLDeserializationContext;

/// The resolver a pipeline-descriptor send borrows for its duration. `Err` is a throw inside the
/// lookup, which leaves `functionWithFunctionRef:` as an Objective-C exception; so does a Rust
/// unwind out of it, which the context keeps ([`DeserializationContext::take_unwound`]).
pub(crate) type Resolver<'a> = dyn FnMut(Hold<'_>, u64) -> RaiseResult<Option<MetalFunction>> + 'a;

/// A send's resolver, borrowed for the send, and the hold of the send's caller, which
/// the resolver runs under.
#[derive(Debug, Clone, Copy)]
struct Installed {
    hold: Hold<'static>,
    resolver: NonNull<Resolver<'static>>,
}

/// The context's state: the borrowed resolver and its caller's hold while a send runs,
/// `None` otherwise; the throw the resolver last raised through Metal, for the send to hand on;
/// and a Rust unwind out of the resolver, for the send to go on with once Metal has returned.
#[derive(Debug, Default)]
pub struct DeserializationContextIvars {
    installed: Cell<Option<Installed>>,
    thrown: RefCell<Option<Raise>>,
    unwound: RefCell<Option<Box<dyn Any + Send>>>,
}

define_class!(
    /// `NSObject <MTLDeserializationContext>`: answers `functionWithFunctionRef:` from the
    /// resource manager's function table. One lives for each pipeline-descriptor send, holding
    /// the send's resolver: Metal calls it synchronously, during the send, on the sending thread,
    /// so the context's lifetime is not observable. The class name is ours (no wire or image
    /// spells it).
    #[unsafe(super(NSObject))]
    #[name = "RVGDeserializationContext"]
    #[ivars = DeserializationContextIvars]
    pub struct DeserializationContext;

    unsafe impl NSObjectProtocol for DeserializationContext {}

    unsafe impl MTLDeserializationContext for DeserializationContext {}

    impl DeserializationContext {
        /// The lookup and its error log are core's resolver (`resourcemanager/resolver.rs`); this
        /// returns what it answers, +0 (autoreleased), or nil.
        ///
        /// A raise inside the lookup leaves the method as an Objective-C exception, raised from
        /// this frame, which owns nothing then (D34): Metal's own behaviour decides whether it
        /// reaches the send. Whichever foreign handler catches it unwinds this frame first, since
        /// objc2 0.6.4 makes every `define_class!` method `extern "C-unwind"` (measured on
        /// macOS 26.7). A Rust unwind out of the lookup never leaves this frame: it is kept, and
        /// the method raises in its place (`resolve`).
        #[unsafe(method(functionWithFunctionRef:))]
        fn function_with_function_ref(&self, function_ref: u64) -> *mut ProtocolObject<dyn MTLFunction> {
            match self.resolve(function_ref) {
                Ok(Some(f)) => Retained::autorelease_return(f.into_retained()),
                Ok(None) => std::ptr::null_mut(),
                Err(exception) => objc2::exception::throw(exception),
            }
        }
    }
);

/// The reason of the exception raised through Metal in place of a Rust unwind (a diagnostic).
const UNWOUND: &str = "the function resolver's thread unwound";

/// `+[NSException exceptionWithName:reason:userInfo:]` of `NSGenericException`, `reason` and no
/// userInfo.
fn generic_exception(reason: &str) -> Retained<Exception> {
    let name = NSString::from_str("NSGenericException");
    let reason = NSString::from_str(reason);
    // SAFETY: a name and a reason, no userInfo.
    let exception =
        unsafe { NSException::exceptionWithName_reason_userInfo(&name, Some(&reason), None) };
    NSException::into_exception(exception)
}

impl DeserializationContext {
    /// A context with no resolver yet.
    pub fn new() -> Retained<DeserializationContext> {
        let this: Allocated<DeserializationContext> = DeserializationContext::alloc();
        let this = this.set_ivars(DeserializationContextIvars::default());
        // SAFETY: `-[NSObject init]` on a freshly allocated object.
        unsafe { msg_send![super(this), init] }
    }

    /// Runs `f` with `resolver` installed under `hold`, the hold of the send's caller:
    /// every `functionWithFunctionRef:` Metal sends during `f` calls it. The resolver is removed
    /// again before this returns, on every path.
    pub fn with_resolver<R>(
        &self,
        hold: Hold<'_>,
        resolver: &mut Resolver<'_>,
        f: impl FnOnce() -> R,
    ) -> R {
        struct Clear<'a>(&'a Cell<Option<Installed>>);
        impl Drop for Clear<'_> {
            fn drop(&mut self) {
                self.0.set(None);
            }
        }
        let ptr: NonNull<Resolver<'_>> = NonNull::from(resolver);
        // SAFETY: only the trait object's lifetime bound changes; the pointer is stored for the
        // duration of `f` alone and cleared by `Clear` before `resolver`'s borrow ends.
        let ptr: NonNull<Resolver<'static>> = unsafe { std::mem::transmute(ptr) };
        // SAFETY: only the lifetime of the hold's holder changes; the hold is stored for the
        // duration of `f` alone, on this thread, and cleared by `Clear` before its borrow ends.
        // `resolve` hands the resolver a hold of its own frame, which cannot outlive it.
        let hold: Hold<'static> = unsafe { std::mem::transmute::<Hold<'_>, Hold<'static>>(hold) };
        self.ivars().installed.set(Some(Installed {
            hold,
            resolver: ptr,
        }));
        let _clear = Clear(&self.ivars().installed);
        f()
    }

    /// Calls the installed resolver; `Ok(None)` when no send is running. The resolver runs under
    /// its caller's hold and answers `RaiseResult`: its raise comes back as `Err` and never
    /// dies inside Metal's send. The raise is kept for [`Self::take_thrown`] and comes back as the
    /// `NSException` to raise through Metal: `+exceptionWithName:reason:userInfo:` of a generic
    /// exception and the raise's detail.
    ///
    /// A Rust unwind out of the resolver (a death in a refusal scope, which unwinds to the scope,
    /// or a test's captured death) is caught here, before any of
    /// Metal's frames, and kept for [`Self::take_unwound`]. Metal gets a generic exception in its
    /// place, which unwinds Metal's frames as a raise's does, and the send goes on with the
    /// unwind once Metal has returned. While one is kept, a later call answers the same
    /// exception and runs nothing.
    #[expect(
        clippy::disallowed_methods,
        reason = "no Rust unwind crosses Metal's frames: the resolver's is carried across (6.3)"
    )]
    fn resolve(&self, function_ref: u64) -> Result<Option<MetalFunction>, Retained<Exception>> {
        if self.ivars().unwound.borrow().is_some() {
            return Err(generic_exception(UNWOUND));
        }
        let Some(Installed {
            hold,
            resolver: mut ptr,
        }) = self.ivars().installed.get()
        else {
            return Ok(None);
        };
        // SAFETY: the pointer is installed only inside `with_resolver`, which holds the unique
        // borrow of the resolver for as long as it is installed; Metal calls back synchronously
        // on the same thread (never re-entrantly into the same resolver).
        let resolver = unsafe { ptr.as_mut() };
        match std::panic::catch_unwind(AssertUnwindSafe(|| resolver(hold, function_ref))) {
            Ok(Ok(f)) => Ok(f),
            Ok(Err(t)) => {
                let exception = generic_exception(t.detail().unwrap_or_default());
                *self.ivars().thrown.borrow_mut() = Some(t);
                Err(exception)
            }
            Err(unwound) => {
                *self.ivars().unwound.borrow_mut() = Some(unwound);
                Err(generic_exception(UNWOUND))
            }
        }
    }

    /// The throw the resolver raised through Metal during the last send, if any: the exception
    /// reaches the caller unchanged.
    pub fn take_thrown(&self) -> Option<Raise> {
        self.ivars().thrown.borrow_mut().take()
    }

    /// The Rust unwind out of the resolver during the last send, if one came: the send's caller
    /// goes on with it (`std::panic::resume_unwind`) once Metal has returned.
    pub fn take_unwound(&self) -> Option<Box<dyn Any + Send>> {
        self.ivars().unwound.borrow_mut().take()
    }

    /// The context as the `id` the SPI takes.
    pub fn as_any(&self) -> &AnyObject {
        self
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use objc2::ClassType;
    use objc2::runtime::AnyClass;
    use reims_vgpu_core::raise::OnFault;

    fn send_ref(ctx: &DeserializationContext, r: u64) -> *mut AnyObject {
        // SAFETY: the method is defined above with this encoding.
        unsafe { msg_send![ctx, functionWithFunctionRef: r] }
    }

    /// Nil outside a send; the resolver's answer and argument inside one
    #[test]
    fn function_with_function_ref_calls_the_resolver() {
        let ctx = DeserializationContext::new();
        assert!(send_ref(&ctx, 1).is_null());
        let mut seen = Vec::new();
        let mut resolver = |_: Hold<'_>, r: u64| {
            seen.push(r);
            Ok(None)
        };
        let got = ctx.with_resolver(Hold::NONE, &mut resolver, || send_ref(&ctx, 0x1_0000_0007));
        assert!(got.is_null());
        assert_eq!(seen, vec![0x1_0000_0007]);
        // Removed after the send.
        assert!(send_ref(&ctx, 2).is_null());
    }

    /// A throw whose first handler is the resolver's boundary is raised through Metal as an
    /// `NSException` of its name and reason, with a nil userInfo, and the Raise kept for the
    /// caller
    #[test]
    fn a_boundary_throw_is_raised_through_metal() {
        let ctx = DeserializationContext::new();
        let mut resolver = |_: Hold<'_>, _: u64| -> RaiseResult<Option<MetalFunction>> {
            Err(reims_vgpu_core::testkit::capture::raise(
                OnFault::Return,
                "resolver",
                "Invalid function reference",
            ))
        };
        let caught = ctx.with_resolver(Hold::NONE, &mut resolver, || {
            objc2::exception::catch(std::panic::AssertUnwindSafe(|| send_ref(&ctx, 3)))
        });
        let exc = caught
            .expect_err("raised through the send")
            .expect("an object");
        let ns = exc.downcast_ref::<NSException>().expect("an NSException");
        assert_eq!(ns.name().to_string(), "NSGenericException");
        assert_eq!(
            ns.reason().map(|r| r.to_string()).as_deref(),
            Some("Invalid function reference")
        );
        assert!(ns.userInfo().is_none());
        let t = ctx.take_thrown().expect("the resolver's Raise");
        assert_eq!(t.detail(), Some("Invalid function reference"));
    }

    /// A death in the resolver inside a refusal scope never unwinds through the send's frames: the
    /// method raises through them as for a throw and keeps the unwind, a later send raises the
    /// same without the resolver, and the death goes on to the scope once the sender takes the
    /// unwind back
    #[test]
    fn a_death_in_the_resolver_is_carried_across_the_send() {
        use reims_vgpu_core::raise::death::{DeathKind, DeathSite};
        use reims_vgpu_core::raise::refusals::refusing;

        let ctx = DeserializationContext::new();
        let mut calls = 0u32;
        let mut resolver = |_: Hold<'_>, _: u64| -> RaiseResult<Option<MetalFunction>> {
            calls += 1;
            reims_vgpu_core::fatal!(DeathSite {
                kind: DeathKind::Abort,
                file: "resolver.test",
                line: 0,
            })
        };
        let mut steps = Vec::new();
        let r = refusing(|| {
            for _ in 0..2 {
                let sent = ctx.with_resolver(Hold::NONE, &mut resolver, || {
                    objc2::exception::catch(std::panic::AssertUnwindSafe(|| send_ref(&ctx, 3)))
                });
                let exc = sent.expect_err("raised through the send");
                let (_, reason) = crate::msg::name_and_reason(exc.as_deref());
                steps.push(reason);
            }
            assert!(ctx.take_thrown().is_none());
            if let Some(unwound) = ctx.take_unwound() {
                std::panic::resume_unwind(unwound)
            }
            steps.push("never".to_owned());
        });
        assert!(r.is_err());
        assert_eq!(calls, 1);
        assert_eq!(steps, vec![UNWOUND, UNWOUND]);
    }

    /// The private classes resolve to the runtime's classes
    #[test]
    fn private_classes_exist() {
        for name in [
            c"MTLBufferDescriptor",
            c"MTLResourceAddressRangeArray",
            c"_MTLIndirectDrawArguments",
        ] {
            assert!(AnyClass::get(name).is_some(), "{name:?}");
        }
        assert_eq!(MTLBufferDescriptor::class().name(), c"MTLBufferDescriptor");
        assert_eq!(
            IndirectDrawArguments::class().name(),
            c"_MTLIndirectDrawArguments"
        );
        assert!(ctx_responds());
    }

    fn ctx_responds() -> bool {
        let ctx = DeserializationContext::new();
        ctx.respondsToSelector(objc2::sel!(functionWithFunctionRef:))
            && ctx.as_any().class().name() == c"RVGDeserializationContext"
    }
}
