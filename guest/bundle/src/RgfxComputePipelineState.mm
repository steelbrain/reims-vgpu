/*
 * RgfxComputePipelineState: see RgfxComputePipelineState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxComputePipelineState.h"
#import "RgfxTexture.h" // rgfxAppleGPU
#include "RgfxLogOnce.hpp"

#import <objc/message.h>

#include <mach-o/dyld.h>
#include <deque>
#include <limits.h>
#include <mutex>
#include <set>
#include <stdlib.h>
#include <string>
#include <vector>

#import "RgfxDynamicLibrary.h"
#import "RgfxFunction.h"
#import "RgfxLibrary.h"
#import "RgfxFunctionTable.h"
#import "RgfxGLFunction.h"
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include "RgfxKnownPipelines.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxSerializer.hpp"

namespace {

// The stage-input descriptor's layout/attribute arrays expose no count, so the mapping scans
// the indices MTLStageInputOutputDescriptor addresses (0..30) and keeps the entries in use: a
// layout whose stride is non-zero, an attribute whose format is not MTLAttributeFormatInvalid.
constexpr NSUInteger kMaxStageInputBuffers = 31;
constexpr NSUInteger kMaxStageInputAttributes = 31;

/// The in-use layouts and attributes of `sd` (nil: none).
void readStageInput(MTLStageInputOutputDescriptor *sd, std::vector<RgfxVertexBufferLayout> &layouts,
                    std::vector<RgfxVertexAttribute> &attributes)
{
	if (sd == nil)
		return;
	for (NSUInteger i = 0; i < kMaxStageInputBuffers; i++) {
		MTLBufferLayoutDescriptor *l = sd.layouts[i];
		if (l.stride == 0)
			continue;
		RgfxVertexBufferLayout rl = {};
		rl.buffer_index = static_cast<uint32_t>(i);
		rl.stride = static_cast<uint32_t>(l.stride);
		rl.step_function = static_cast<uint32_t>(l.stepFunction);
		rl.step_rate = static_cast<uint32_t>(l.stepRate);
		layouts.push_back(rl);
	}
	for (NSUInteger i = 0; i < kMaxStageInputAttributes; i++) {
		MTLAttributeDescriptor *a = sd.attributes[i];
		if (a.format == MTLAttributeFormatInvalid)
			continue;
		RgfxVertexAttribute ra = {};
		ra.attribute_index = static_cast<uint32_t>(i);
		ra.format = static_cast<uint32_t>(a.format);
		ra.offset = static_cast<uint32_t>(a.offset);
		ra.buffer_index = static_cast<uint32_t>(a.bufferIndex);
		attributes.push_back(ra);
	}
}

/// The refusal of a pipeline that names `fn`, a function another device's library made: it
/// names no library the host holds.
NSError *foreignFunctionError(NSString *what, id<MTLFunction> fn)
{
	NSString *msg = [NSString stringWithFormat:@"compute pipeline: %@ %@ (%@) is not from a library of this device",
	                                           what, fn.name, NSStringFromClass([(id)fn class])];
	static char site;
	if (rgfxLogFirst(&site))
		NSLog(@"RGFX: %@", msg);
	return RgfxMakeError(RgfxMetalErrorInvalidDescriptor, msg);
}

/// The path of the file install name `name` names, read as dyld reads one: an absolute path as
/// it is, `@executable_path/` from the process's executable's directory, and `@loader_path/` from
/// the directory of the file the library naming it was read from (`loader`). Empty for any other
/// name (relative, `@rpath/`), and for `@loader_path/` of a URL that is no file URL: they name no
/// file here.
std::string installNamePath(const std::string &name, NSURL *loader)
{
	auto under = [&](const std::string &directory, size_t prefix) {
		return directory.empty() ? std::string() : directory + "/" + name.substr(prefix);
	};
	if (!name.empty() && name[0] == '/')
		return name;
	static const std::string executable = "@executable_path/";
	static const std::string loaderPath = "@loader_path/";
	if (name.compare(0, executable.size(), executable) == 0) {
		char path[PATH_MAX];
		uint32_t size = sizeof(path);
		char real[PATH_MAX];
		if (_NSGetExecutablePath(path, &size) != 0 || realpath(path, real) == nullptr)
			return std::string();
		std::string directory(real);
		directory.erase(directory.find_last_of('/'));
		return under(directory, executable.size());
	}
	if (name.compare(0, loaderPath.size(), loaderPath) == 0 && loader.fileURL) {
		NSString *directory = loader.URLByDeletingLastPathComponent.path;
		return under(directory.fileSystemRepresentation, loaderPath.size());
	}
	return std::string();
}

/// The x86 Macs' refusal of a kernel whose libraries name an install name no file loads, or that
/// a library made of no file names (measured: the AMD Radeon Pro 5300M and the Intel UHD Graphics
/// 630, macOS 26.7.1, oracle_fnptr's "unresolved bare", "linked data bare" and "source"; beside a
/// preloaded library that exports the symbol, "unresolved preloaded", "linked data preloaded" and
/// "source preloaded", the 5300M refuses it too and the 630 makes the pipeline: the bundle answers
/// the 5300M's).
NSError *unresolvedLibraryError(const std::string &name)
{
	static char site;
	if (rgfxLogFirst(&site))
		NSLog(@"RGFX: compute pipeline: the dynamic library %s loads from no file", name.c_str());
	return [NSError errorWithDomain:@"CompilerError"
	                           code:2
	                       userInfo:@{NSLocalizedDescriptionKey : @"Error creating compute variant"}];
}

/// The array `object` answers to the first of `names` it responds to, nil where none: members a
/// guest's Metal may not have (preloadedLibraries is insertLibraries before macOS 13, and
/// MTLLinkedFunctions' privateFunctions is macOS 12's).
NSArray *arrayOf(id object, std::initializer_list<const char *> names)
{
	for (const char *name : names) {
		SEL sel = sel_registerName(name);
		if ([object respondsToSelector:sel])
			return reinterpret_cast<NSArray *(*)(id, SEL)>(objc_msgSend)(object, sel);
	}
	return nil;
}

/// The dynamic libraries a compute pipeline's kernel links into `refs` (their install names kept
/// in `names`, the objects in `held`): the descriptor's preloadedLibraries in order, then for each
/// install name the kernel's library, its linked functions' libraries and these dynamic libraries
/// name, in turn and once each, the library of the file it names (installNamePath), read here as
/// Metal reads it when it creates the pipeline (at each creation, so a replaced file links its new
/// code), and its own install names after. A name loads from its file whether or not a preloaded
/// library exports what it is linked for, as Metal's creation needs it to (the runtime's
/// DynamicLibraries, read from code: measured on an M3 Max and an M4; the AMD Radeon Pro 5300M
/// refuses a kernel whose name loads nothing beside a preloaded library that exports its symbol,
/// oracle_fnptr's "unresolved preloaded"). Metal loads the names of a library made from a file
/// alone: a name that a library made of bytes or of source names refuses the pipeline, where its
/// file is there too (measured: the 5300M and the Intel UHD Graphics 630, macOS 26.7.1,
/// oracle_fnptr's "linked data bare" and "source", where "by install name", the same bytes read
/// from their file, makes it). On an Apple GPU (rgfxAppleGPU) a library made of bytes or of source
/// loads its @executable_path names too: the M3 Max makes both those pipelines, bare and
/// preloaded, and they run as the file's (measured: macOS 27.0.1, oracle_fnptr Apple); its
/// @loader_path names, which no file is the loader of, still refuse it (not measured). False and
/// the error where a preloaded library is not one of ours, a library of no file names an install
/// name (but as above), or a name loads no dynamic library, as none does where the host makes
/// none: the host is never left to load one from a file of its own.
bool pipelineLibraries(MTLComputePipelineDescriptor *descriptor, RgfxFunction *kernel,
                       id<MTLDevice> device, NSMutableArray *held,
                       std::vector<RgfxPipelineLibraryRef> &refs, std::deque<std::string> &names,
                       NSError *__autoreleasing *error)
{
	// On an Apple GPU a library of no file loads its names from the executable's directory too.
	const bool apple = rgfxAppleGPU(device);
	static const std::string executable = "@executable_path/";
	std::set<std::string> loadedNames;
	std::deque<std::pair<std::string, NSURL *>> pending;
	auto dependenciesOf = [&](RgfxResource *resource) {
		if (![resource isKindOfClass:[RgfxLibrary class]])
			return;
		RgfxLibrary *library = (RgfxLibrary *)resource;
		for (std::string &name : [library rgfxDependencies])
			pending.emplace_back(std::move(name), [library rgfxURL]);
	};
	for (id<MTLDynamicLibrary> preloaded in
	     arrayOf(descriptor, {"preloadedLibraries", "insertLibraries"})) {
		if (![(id)preloaded isKindOfClass:[RgfxDynamicLibrary class]]) {
			if (error)
				*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
				                       @"compute pipeline: a preloaded library is not of this "
				                       @"device");
			return false;
		}
		RgfxDynamicLibrary *library = (RgfxDynamicLibrary *)preloaded;
		refs.push_back({library.rgfxObjId, nullptr});
		[held addObject:library];
	}
	dependenciesOf([kernel rgfxLibrary]);
	MTLLinkedFunctions *linked = descriptor.linkedFunctions;
	for (NSArray<id<MTLFunction>> *list in @[
		     linked.functions ?: @[], linked.binaryFunctions ?: @[],
		     arrayOf(linked, {"privateFunctions"}) ?: @[]
	     ])
		for (id<MTLFunction> f in list)
			if ([(id)f isKindOfClass:[RgfxFunction class]])
				dependenciesOf([(RgfxFunction *)f rgfxLibrary]);
	for (NSArray<id<MTLFunction>> *group in linked.groups.allValues)
		for (id<MTLFunction> f in group)
			if ([(id)f isKindOfClass:[RgfxFunction class]])
				dependenciesOf([(RgfxFunction *)f rgfxLibrary]);
	for (RgfxDynamicLibrary *preloaded in [held copy])
		dependenciesOf(preloaded.rgfxLibrary);
	while (!pending.empty()) {
		auto [name, loader] = std::move(pending.front());
		pending.pop_front();
		if (loader == nil && !(apple && name.compare(0, executable.size(), executable) == 0)) {
			if (error)
				*error = unresolvedLibraryError(name);
			return false;
		}
		if (!loadedNames.insert(name).second)
			continue;
		const std::string path = installNamePath(name, loader);
		NSURL *url = path.empty() ? nil
		                          : [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
		id<MTLDynamicLibrary> loaded =
		    url != nil && device != nil ? [device newDynamicLibraryWithURL:url error:nil] : nil;
		if (![(id)loaded isKindOfClass:[RgfxDynamicLibrary class]]) {
			if (error)
				*error = unresolvedLibraryError(name);
			return false;
		}
		RgfxDynamicLibrary *library = (RgfxDynamicLibrary *)loaded;
		names.push_back(name);
		refs.push_back({library.rgfxObjId, names.back().c_str()});
		[held addObject:library];
		dependenciesOf(library.rgfxLibrary);
	}
	return true;
}

// RgfxComputeProgressError of the descriptor's forwardProgressUsage and contentionRelief, where its
// Metal (macOS 27) has them.
NSError *progressErrorOf(MTLComputePipelineDescriptor *descriptor)
{
	SEL progress = sel_registerName("forwardProgressUsage"), relief = sel_registerName("contentionRelief");
	const BOOL hasProgress = [descriptor respondsToSelector:progress];
	const BOOL hasRelief = [descriptor respondsToSelector:relief];
	return RgfxComputeProgressError(
	    hasProgress, hasProgress ? ((NSInteger (*)(id, SEL))objc_msgSend)(descriptor, progress) : 0, hasRelief,
	    hasRelief ? ((NSInteger (*)(id, SEL))objc_msgSend)(descriptor, relief) : 0);
}

// What Metal's compiler answers a refusal of a pipeline's threadgroup fields with: the x86 Macs'
// CompilerError 2, the M3 Max's AGXMetalG15X_M1 3 (oracle_threads, below).
NSError *threadsError(RgfxThreadsRules rules, NSString *text)
{
	const bool apple = rules == RgfxThreadsRulesApple;
	return [NSError errorWithDomain:apple ? @"AGXMetalG15X_M1" : @"CompilerError"
	                           code:apple ? 3 : 2
	                       userInfo:@{NSLocalizedDescriptionKey : text}];
}

} // namespace

@implementation RgfxThreadsVerdict {
@public
	NSString *_assertion;
	const char *_assertionFunction;
	NSError *_error;
	NSUInteger _maxTotalThreads;
}

- (NSString *)assertion
{
	return _assertion;
}

- (const char *)assertionFunction
{
	return _assertionFunction;
}

- (NSError *)error
{
	return _error;
}

- (NSUInteger)maxTotalThreads
{
	return _maxTotalThreads;
}

@end

// A compute pipeline's threadgroup fields as Metal checks them (measured: oracle_threads, each step
// a first compile of its kernel, plain and with MTL_DEBUG_LAYER=1 alike, on the AMD Radeon Pro
// 5300M and the Intel UHD Graphics 630 under macOS 26.7.1 (25G241), and the M3 Max under macOS
// 27.0.1 (26A434)). On all three, a required size (`requiredThreadsPerThreadgroup`, not {0, 0, 0})
// with an axis of 0, of more threads than the device's limit, or of more than the descriptor's
// `maxTotalThreadsPerThreadgroup` ends the process at its validation's failed assertion
// (validateWithDevice:1530 on the x86 Macs, :1623 on the M3 Max), each failed check a line of its
// words. Then, in the x86 Macs' order (the 5300M's words; the 630's below) or the M3 Max's:
// - a `maxTotalThreadsPerThreadgroup` past the device's limit is refused (x86 first; the M3 Max
//   last, as a kernel's attribute it differs from is refused first; the 630 makes the pipeline);
// - a required size of more threads than the kernel's [[max_total_threads_per_threadgroup(n)]] is
//   refused, in the same words on all three;
// - a `maxTotalThreadsPerThreadgroup` other than what the kernel specifies, the required size's
//   threads where it has one, else its attribute's n, is refused (the 630: "[[max_total_threads_per_
//   threadgroup]] declared in Metal doesn't match [MTLComputePipelineDescriptor
//   maxTotalThreadsPerThreadgroup]").
// A pipeline made of a required size answers its threads as its maxTotalThreadsPerThreadgroup on
// the 5300M and the M3 Max (the 630 answers 1024 throughout, and 448 for one thread). A kernel
// compiled before is made of a required size past its attribute on all three, answering the
// attribute (oracle_threads' -warm steps): Metal's compile cache, which this does not follow, as
// the runtime does not (its RequiredThreadsProblem::OverKernelMax).
RgfxThreadsVerdict *RgfxComputeThreadsVerdictOf(RgfxThreadsRules rules, MTLSize required, NSUInteger maxTotal,
                                                NSUInteger kernelMax, NSString *kernelName, NSUInteger deviceLimit)
{
	RgfxThreadsVerdict *v = [RgfxThreadsVerdict new];
	const bool apple = rules == RgfxThreadsRulesApple;
	const bool hasRequired = required.width != 0 || required.height != 0 || required.depth != 0;
	const NSUInteger threads = required.width * required.height * required.depth;
	if (hasRequired) {
		NSString *size = [NSString
		    stringWithFormat:@"(requiredThreadsPerThreadgroup.width(%lu) * requiredThreadsPerThreadgroup.height(%lu) * "
		                     @"requiredThreadsPerThreadgroup.depth(%lu))(%lu)",
		                     static_cast<unsigned long>(required.width), static_cast<unsigned long>(required.height),
		                     static_cast<unsigned long>(required.depth), static_cast<unsigned long>(threads)];
		NSMutableString *failed = [NSMutableString string];
		if (threads == 0)
			[failed appendFormat:@"%@ must not be 0.\n", size];
		if (deviceLimit != 0 && threads > deviceLimit)
			[failed appendFormat:@"%@ must be <= %lu (device limit)\n", size, static_cast<unsigned long>(deviceLimit)];
		if (maxTotal != 0 && threads > maxTotal)
			[failed appendFormat:@"%@ must be <= maxTotalThreadsPerThreadgroup (%lu)\n", size,
			                     static_cast<unsigned long>(maxTotal)];
		if (failed.length != 0) {
			v->_assertion = [@"Compute Pipeline Descriptor Validation\n" stringByAppendingString:failed];
			v->_assertionFunction = apple ? "validateWithDevice:1623" : "validateWithDevice:1530";
			return v;
		}
	}
	NSString *pastLimit =
	    deviceLimit != 0 && maxTotal > deviceLimit
	        ? [NSString stringWithFormat:apple ? @": Specified total max threads per threadgroup (%lu) exceeds the "
	                                             @"maximum total threads per threadgroup supported (%lu)"
	                                           : @"maxThreadsPerThreadgroup pipeline descriptor value (%lu) exceeds "
	                                             @"maximum supported (%lu)\n",
	                                     static_cast<unsigned long>(maxTotal), static_cast<unsigned long>(deviceLimit)]
	        : nil;
	if (!apple && pastLimit != nil) {
		v->_error = threadsError(rules, pastLimit);
		return v;
	}
	if (hasRequired && kernelMax != 0 && threads > kernelMax) {
		v->_error = threadsError(
		    rules, [NSString stringWithFormat:@"error: requested required threads per threadgroup %lu,%lu,%lu but '%@' "
		                                      @"expects max threads per threadgroup %lu\n",
		                                      static_cast<unsigned long>(required.width),
		                                      static_cast<unsigned long>(required.height),
		                                      static_cast<unsigned long>(required.depth), kernelName,
		                                      static_cast<unsigned long>(kernelMax)]);
		return v;
	}
	const NSUInteger specified = hasRequired ? threads : kernelMax;
	if (maxTotal != 0 && specified != 0 && specified != maxTotal) {
		v->_error = threadsError(
		    rules, [NSString stringWithFormat:apple ? @": Kernel specified max total threads per threadgroup (%lu) must "
		                                              @"match compute pipeline specified max total threads per "
		                                              @"threadgroup (%lu)"
		                                            : @"max_total_threads_per_threadgroup shader attribute(Metadata) "
		                                              @"value (%lu) does not match maxThreadsPerThreadgroup pipeline "
		                                              @"descriptor value (%lu)\n",
		                                      static_cast<unsigned long>(specified),
		                                      static_cast<unsigned long>(maxTotal)]);
		return v;
	}
	if (pastLimit != nil) {
		v->_error = threadsError(rules, pastLimit);
		return v;
	}
	v->_maxTotalThreads = hasRequired ? threads : 0;
	return v;
}

// The M3 Max under macOS 27.0.1 makes a pipeline of each forwardProgressUsage Metal names
// (Automatic 0, Weak 1, SIMDGroupParallel 2) and contentionRelief (Automatic 0, None 1); of any
// other value its compiler service ends and the pipeline is refused (oracle_threads' -fp and -cr
// steps). The x86 Macs' Metal has neither field (macOS 26).
NSError *RgfxComputeProgressError(BOOL hasProgress, NSInteger progress, BOOL hasRelief, NSInteger relief)
{
	if ((hasProgress && (progress < 0 || progress > 2)) || (hasRelief && (relief < 0 || relief > 1)))
		return [NSError errorWithDomain:@"AGXMetalG15X_M1"
		                           code:2
		                       userInfo:@{
			                       NSLocalizedDescriptionKey :
			                           @"Compilation failed due to an interrupted connection: "
			                           @"XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries."
		                       }];
	return nil;
}

// Defined in the main implementation; declared here so the lambda in
// rgfxRegisterWithReplyReader: sees its signature.
@interface RgfxComputePipelineState ()
- (void)rgfxFillReflection:(const RgfxPipelineReflection *)r;
// A create posted without waiting takes `known`, the equal create's reflection, at once, and
// later its own answer.
- (void)rgfxTakeKnown:(const RgfxPipelineReflection &)known;
- (void)rgfxTakeAnswer:(const RgfxPipelineReflection *)r;
// Keeps what the descriptor asked that the pipeline answers (rgfx*Desc ivars).
- (void)rgfxTakeDescriptor:(MTLComputePipelineDescriptor *)descriptor flags:(uint32_t)flags;
@end

@implementation RgfxComputePipelineState {
	uint64_t _seqno;
	// What the create asked that the pipeline answers: its RGFX_COMPUTE_PIPELINE_* flags, the
	// required threadgroup size and the forward-progress use (Metal's defaults for a pipeline
	// made from a function alone).
	uint32_t _createFlags;
	MTLSize _requiredThreadsPerThreadgroup;
	NSInteger _forwardProgressUsage;
	// The required size's threads, which maxTotalThreadsPerThreadgroup answers (0: none required).
	NSUInteger _requiredThreads;
	// The host's reflection, filled once RGFX_EVT_PIPELINE_REFLECTION for `_seqno`
	// arrives; until then `_haveReflection` is NO and the getters answer placeholders.
	RgfxPipelineReflection _reflection;
	BOOL _haveReflection;
	std::shared_ptr<const RgfxAirFunction> _air; // its function's arguments; null where unknown
	// A keyed create's key, and the known pipelines its answer is kept in once accepted, until
	// it registers; a create posted without waiting has its own answer to wait for.
	std::shared_ptr<RgfxKnownPipelines> _known;
	RgfxPipelineKey _key;
	BOOL _deferred;
	RgfxPipelineAnswer _answer;
	NSInteger _textureWriteRoundingMode; // the descriptor's (MTLComputePipelineDescriptorSPI)
	// functionHandleWithFunction:'s answers, by function (NSNull for nil), under _handlesLock.
	std::mutex _handlesLock;
	NSMapTable *_handles;
	// The dynamic libraries its create named, kept with it.
	NSArray *_libraries;
}

// The host drops the pipeline as the app lets it go, after any command buffer that names it (each
// keeps it while it runs).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

// A pipeline of the function alone is one of a descriptor of it, every other field Metal's
// default. Metal's descriptor takes no nil function, so a nil one is refused here.
+ (instancetype)computePipelineStateWithFunction:(id<MTLFunction>)function
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
                                           error:(__autoreleasing NSError **)error
{
	return [self computePipelineStateWithFunction:function
	                                       device:device
	                                   serializer:serializer
	                                        known:nullptr
	                                        error:error];
}

+ (instancetype)computePipelineStateWithFunction:(id<MTLFunction>)function
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
                                           known:(std::shared_ptr<RgfxKnownPipelines>)known
                                           error:(__autoreleasing NSError **)error
{
	if (function == nil) {
		if (error)
			*error = RgfxMakeError(serializer == nullptr ? RgfxMetalErrorNoRoute
			                                             : RgfxMetalErrorInvalidDescriptor,
			                       serializer == nullptr
			                           ? @"compute pipeline: no serialiser route"
			                           : @"compute pipeline: no compute function of this device");
		return nil;
	}
	MTLComputePipelineDescriptor *descriptor = [[MTLComputePipelineDescriptor alloc] init];
	descriptor.computeFunction = function;
	return [self computePipelineStateWithDescriptor:descriptor
	                                         device:device
	                                     serializer:serializer
	                                          known:std::move(known)
	                                          error:error];
}

+ (instancetype)computePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            device:(id<MTLDevice>)device
                                        serializer:(RgfxSerializer *)serializer
                                             error:(__autoreleasing NSError **)error
{
	return [self computePipelineStateWithDescriptor:descriptor
	                                         device:device
	                                     serializer:serializer
	                                          known:nullptr
	                                          error:error];
}

+ (instancetype)computePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            device:(id<MTLDevice>)device
                                        serializer:(RgfxSerializer *)serializer
                                             known:(std::shared_ptr<RgfxKnownPipelines>)known
                                             error:(__autoreleasing NSError **)error
{
	if (serializer == nullptr) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute, @"compute pipeline: no serialiser route");
		return nil;
	}
	// A GL kernel variant that takes a patch control point input, which Metal compiles in no
	// kernel: the pipeline is one of the variant made a kernel under the stage input descriptor,
	// which takes no stage input (RgfxGLStageInKernel). The device's creators answer the
	// reflection of the descriptor's own function and stage input descriptor, as for any other.
	id<MTLFunction> stageInKernel = nil;
	NSError *stageInRefusal = nil;
	if (RgfxGLStageInKernel(descriptor.computeFunction, descriptor.stageInputDescriptor, &stageInKernel,
	                        &stageInRefusal)) {
		if (stageInKernel == nil) {
			if (error)
				*error = stageInRefusal;
			return nil;
		}
		MTLComputePipelineDescriptor *kernelDescriptor = [descriptor copy];
		kernelDescriptor.computeFunction = stageInKernel;
		kernelDescriptor.stageInputDescriptor = nil;
		return [self computePipelineStateWithDescriptor:kernelDescriptor
		                                         device:device
		                                     serializer:serializer
		                                          known:std::move(known)
		                                          error:error];
	}
	id<MTLFunction> function = descriptor.computeFunction;
	if (NSError *unspecialised = [RgfxFunction rgfxUnspecialisedError:function role:@"function"]) {
		if (error)
			*error = unspecialised;
		return nil;
	}
	if (function == nil) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                       @"compute pipeline: no compute function of this device");
		return nil;
	}
	if (![(id)function isKindOfClass:[RgfxFunction class]]) {
		if (error)
			*error = foreignFunctionError(@"function", function);
		return nil;
	}
	RgfxLinkedFunctionList linked;
	if (id<MTLFunction> foreign = RgfxCollectLinkedFunctions(descriptor.linkedFunctions, linked)) {
		if (error)
			*error = foreignFunctionError(@"linked function", foreign);
		return nil;
	}

	RgfxFunction *rf = (RgfxFunction *)function;
	// Its threadgroup fields and macOS 27's progress fields, as Metal checks them before it compiles
	// (RgfxComputeThreadsVerdictOf, RgfxComputeProgressError): the host is sent none of them.
	MTLSize required = MTLSizeMake(0, 0, 0);
	if ([descriptor respondsToSelector:@selector(requiredThreadsPerThreadgroup)])
		required = descriptor.requiredThreadsPerThreadgroup;
	std::shared_ptr<const RgfxAirFunction> kernelAir = [rf rgfxAirFunction];
	const char *entryName = [rf rgfxEntryPointUTF8];
	RgfxThreadsVerdict *threads = RgfxComputeThreadsVerdictOf(
	    RgfxGuestThreadsRules(), required, descriptor.maxTotalThreadsPerThreadgroup,
	    kernelAir != nullptr ? kernelAir->maxTotalThreads : 0, entryName != nullptr ? @(entryName) : @"",
	    [device respondsToSelector:@selector(maxThreadsPerThreadgroup)] ? device.maxThreadsPerThreadgroup.width : 0);
	if (threads.assertion != nil)
		RgfxMetalAssertionFailed(threads.assertionFunction, threads.assertion);
	NSError *refusal = threads.error ?: progressErrorOf(descriptor);
	if (refusal != nil) {
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: compute pipeline of %s refused, as Metal refuses it: %@", entryName ?: "",
			      refusal.localizedDescription);
		if (error)
			*error = refusal;
		return nil;
	}
	RgfxComputePipelineParams p = {};
	p.function.library = rf.rgfxLibraryObjId;
	p.function.entry_point = [rf rgfxEntryPointUTF8] ? [rf rgfxEntryPointUTF8] : "";
	p.function.constants = [rf rgfxConstants];
	p.function.constant_count = [rf rgfxConstantCount];
	if (descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth)
		p.flags |= RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH;
	if (descriptor.supportIndirectCommandBuffers)
		p.flags |= RGFX_COMPUTE_PIPELINE_SUPPORT_ICB;
	// A kernel's function of OpenCL mode (newFunctionWithPluginData:bitcodeType:, made only where
	// the host makes such pipelines): the host links it for OpenCL mode, and answers the maximum
	// threads per threadgroup of one of reqd_work_group_size.
	if ([rf rgfxClBindings] != nullptr)
		p.flags |= RGFX_COMPUTE_PIPELINE_OPENCL_MODE;
	p.max_total_threads_per_threadgroup =
	    static_cast<uint32_t>(descriptor.maxTotalThreadsPerThreadgroup);
	p.max_call_stack_depth = static_cast<uint32_t>(descriptor.maxCallStackDepth);

	std::vector<RgfxVertexBufferLayout> layouts;
	std::vector<RgfxVertexAttribute> attributes;
	MTLStageInputOutputDescriptor *stageInput = descriptor.stageInputDescriptor;
	readStageInput(stageInput, layouts, attributes);
	p.stage_input_layouts = layouts.empty() ? nullptr : layouts.data();
	p.stage_input_layout_count = static_cast<uint32_t>(layouts.size());
	p.stage_input_attributes = attributes.empty() ? nullptr : attributes.data();
	p.stage_input_attribute_count = static_cast<uint32_t>(attributes.size());
	if (stageInput != nil) {
		p.stage_input_index_type = static_cast<uint32_t>(stageInput.indexType);
		p.stage_input_index_buffer_index = static_cast<uint32_t>(stageInput.indexBufferIndex);
	}
	p.linked = linked.refs.empty() ? nullptr : linked.refs.data();
	p.linked_count = static_cast<uint32_t>(linked.refs.size());
	std::vector<RgfxPipelineLibraryRef> libraries;
	std::deque<std::string> installNames;
	NSMutableArray *heldLibraries = [NSMutableArray array];
	if (!pipelineLibraries(descriptor, rf, device, heldLibraries, libraries, installNames, error))
		return nil;
	p.libraries = libraries.empty() ? nullptr : libraries.data();
	p.library_count = static_cast<uint32_t>(libraries.size());

	RgfxPipelinePost post;
	post.known = known.get();
	RgfxCreated created = serializer->createComputePipeline(p, &post);
	if (!created.posted) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRingFull,
			                       @"compute pipeline: create did not fit the ring");
		return nil;
	}
	if (error)
		*error = nil;
	RgfxComputePipelineState *state =
	    [[RgfxComputePipelineState alloc] initWithObjId:created.obj_id
	                                              seqno:created.seqno
	                                              label:descriptor.label
	                                             device:device
	                                         serializer:serializer];
	[state rgfxTakeDescriptor:descriptor flags:p.flags];
	if (state != nil) {
		state->_requiredThreads = threads.maxTotalThreads;
		state->_air = [rf rgfxAirFunction];
		state->_libraries = [heldLibraries copy];
		if (post.deferred) {
			[state rgfxTakeKnown:post.reflection];
		} else if (post.keyed) {
			state->_known = known;
			state->_key = post.key;
		}
	}
	return state;
}

- (void)rgfxTakeKnown:(const RgfxPipelineReflection &)known
{
	_reflection = known;
	_reflection.header.seqno = _seqno;
	_reflection.obj_id = self.rgfxObjId;
	_reflection.gpu_resource_id = 0; // its own, which gpuResourceID waits for
	_haveReflection = YES;
	_deferred = YES;
}

- (BOOL)rgfxDeferred
{
	return _deferred;
}

- (std::shared_ptr<const RgfxAirFunction>)rgfxAirFunction
{
	return _air;
}

- (void)rgfxTakeDescriptor:(MTLComputePipelineDescriptor *)descriptor flags:(uint32_t)flags
{
	_createFlags = flags;
	// Both are newer than some guests' Metal (macOS 26 and 27): absent, Metal's defaults stand.
	if ([descriptor respondsToSelector:@selector(requiredThreadsPerThreadgroup)])
		_requiredThreadsPerThreadgroup = descriptor.requiredThreadsPerThreadgroup;
	// Automatic (0) is answered Weak (1), as the M3 Max under macOS 27.0.1 answers it, made from a
	// function alone too (oracle_threads' plain-fn and -fp steps).
	if ([descriptor respondsToSelector:@selector(forwardProgressUsage)]) {
		_forwardProgressUsage = static_cast<NSInteger>(descriptor.forwardProgressUsage);
		if (_forwardProgressUsage == 0)
			_forwardProgressUsage = 1;
	}
	// How the pipeline's texture writes round floats, which Metal's descriptor keeps (0, its
	// default, where the guest's Metal has no such member) and its pipeline answers. The host
	// rounds as its default.
	SEL rounding = sel_registerName("textureWriteRoundingMode");
	if ([descriptor respondsToSelector:rounding])
		_textureWriteRoundingMode = ((NSInteger (*)(id, SEL))objc_msgSend)(descriptor, rounding);
	if (_textureWriteRoundingMode != 0)
		RgfxLogUnsupported([self class], rounding,
		                   "a texture write rounding mode other than the default (none on the wire)");
}

- (instancetype)initWithObjId:(uint64_t)objId
                        seqno:(uint64_t)seqno
                        label:(NSString *)label
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super initWithObjId:objId device:device serializer:serializer];
	if (self) {
		_seqno = seqno;
		self.label = label;
	}
	return self;
}

- (uint64_t)rgfxSeqno
{
	return _seqno;
}

- (void)rgfxRegisterWithReplyReader:(RgfxReplyReader *)reader
{
	if (reader == nullptr) {
		return;
	}
	__weak RgfxComputePipelineState *weakSelf = self;
	if (_deferred) {
		// Its own answer: the resource ID gpuResourceID waits for.
		reader->expectReflection(_seqno, [weakSelf](const RgfxPipelineReflection &r) {
			RgfxComputePipelineState *strongSelf = weakSelf;
			if (strongSelf != nil)
				[strongSelf rgfxTakeAnswer:&r];
		});
		return;
	}
	// A keyed create's answer is kept once the device accepted it, so equal creates need not wait.
	std::shared_ptr<RgfxKnownPipelines> known = std::move(_known);
	const RgfxPipelineKey key = _key;
	reader->expectReflection(_seqno, [weakSelf, known, key](const RgfxPipelineReflection &r) {
		if (known)
			known->learn(key, r);
		RgfxComputePipelineState *strongSelf = weakSelf;
		if (strongSelf != nil) {
			[strongSelf rgfxFillReflection:&r];
		}
	});
}

// A create posted without waiting: its own answer, whose resource ID gpuResourceID answers (as
// RgfxRenderPipelineState's).
- (void)rgfxTakeAnswer:(const RgfxPipelineReflection *)r
{
	_answer.take(*r);
	static char site;
	if (r->status != RGFX_PIPELINE_OK && rgfxLogFirst(&site, r->status))
		NSLog(@"RGFX: the device refused compute pipeline %llu, made as an equal one it accepted",
		      static_cast<unsigned long long>(self.rgfxObjId));
}

- (void)rgfxFillReflection:(const RgfxPipelineReflection *)r
{
	_reflection = *r;
	_haveReflection = YES;
}

- (BOOL)rgfxRefused
{
	return _haveReflection && _reflection.status == RGFX_PIPELINE_REFUSED;
}

// --- MTLComputePipelineState: reflection --------------------------------------
// The host's RgfxPipelineReflection reply is read by the reply reader. Before it
// arrives these answer documented placeholders; after it they report the host's
// values.

- (NSUInteger)maxTotalThreadsPerThreadgroup
{
	// A required size's threads, as Metal's pipelines of one answer (RgfxComputeThreadsVerdictOf).
	if (_requiredThreads != 0)
		return _requiredThreads;
	if (_haveReflection) {
		return static_cast<NSUInteger>(_reflection.max_total_threads_per_threadgroup);
	}
	return 1024; // Metal's common threadgroup floor until reflection is read
}

- (NSUInteger)threadExecutionWidth
{
	if (_haveReflection) {
		return _reflection.thread_execution_width;
	}
	return 32; // the common Apple-GPU SIMD width until reflection is read
}

- (NSUInteger)staticThreadgroupMemoryLength
{
	if (_haveReflection) {
		return static_cast<NSUInteger>(_reflection.static_threadgroup_memory_length);
	}
	return 0;
}

- (MTLResourceID)gpuResourceID
{
	if (_deferred) {
		RgfxSerializer *serializer = [self serializer];
		const uint64_t seqno = _seqno;
		return (MTLResourceID){ _answer.resourceId([serializer, seqno] {
			if (serializer != nullptr)
				serializer->awaitAnswer(seqno);
		}) };
	}
	if (_haveReflection) {
		return (MTLResourceID){ _reflection.gpu_resource_id };
	}
	return (MTLResourceID){0}; // 0 until the reflection reply carries the real id
}

// --- what the create asked ----------------------------------------------------------

- (BOOL)supportIndirectCommandBuffers
{
	if (_haveReflection)
		return (_reflection.flags & RGFX_PIPELINE_REFLECT_SUPPORT_ICB) != 0;
	return (_createFlags & RGFX_COMPUTE_PIPELINE_SUPPORT_ICB) != 0;
}

- (MTLSize)requiredThreadsPerThreadgroup
{
	return _requiredThreadsPerThreadgroup;
}

- (MTLForwardProgressUsage)forwardProgressUsage
{
	return static_cast<MTLForwardProgressUsage>(_forwardProgressUsage);
}

// The device validates no shader, so no pipeline of it runs with validation.
- (MTLShaderValidation)shaderValidation
{
	return MTLShaderValidationDisabled;
}

// Metal's answer for every pipeline a device (not a Metal 4 compiler) makes.
- (MTLComputePipelineReflection *)reflection
{
	return nil;
}

// --- Metal's private members, as its pipelines answer them (measured: macOS 27, M3 Max) ---

// 0 for every pipeline, one that supports indirect command buffers too.
- (uint64_t)gpuAddress
{
	return 0;
}

// Unique and stable for the object while it lives: Metal's is an address inside its pipeline
// object, ours the object's address.
- (uint64_t)uniqueIdentifier
{
	return static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)self));
}

// --- MTLComputePipelineStateSPI ------------------------------------------------------------
// The members of Metal's compute-pipeline SPI that the AMD 5300M's and the Intel 630's pipelines
// both answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m),
// each as Metal's base pipeline, _MTLComputePipelineState, answers it (read from code, macOS
// 27.0.1).

// Metal's base pipeline answers gpuResourceID with gpuHandle, and gpuHandle with resourceIndex: so
// these answer gpuResourceID's value, the one the host numbered the pipeline by.
- (uint64_t)resourceIndex
{
	return [self gpuResourceID]._impl;
}

- (uint64_t)gpuHandle
{
	return [self resourceIndex];
}

// The low 32 bits of uniqueIdentifier.
- (uint32_t)getComputeKernelTelemetryID
{
	return static_cast<uint32_t>([self uniqueIdentifier]);
}

// No debug instrumentation, global constants or binaries: the device instruments no shader, and
// the base answers these for every pipeline.
- (id)debugInstrumentationData
{
	return nil;
}

- (uint64_t)globalConstantsBase
{
	return 0;
}

- (uint64_t)globalConstantsSize
{
	return 0;
}

- (NSDictionary *)pipelineBinaries
{
	return @{};
}

// MTLShaderValidationDisabled, as shaderValidation answers.
- (NSInteger)shaderValidationState
{
	return MTLShaderValidationDisabled;
}

// The descriptor's rounding of floats on texture writes, as the base keeps it.
- (NSInteger)textureWriteRoundingMode
{
	return _textureWriteRoundingMode;
}

// The base raises an unrecognized selector for this (it sends itself doesNotRecognizeSelector:),
// and so does this.
- (id)functionReflectionWithFunctionDescriptor:(id)descriptor
{
	(void)descriptor;
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

// --- not on the wire ------------------------------------------------------------------
// The host answers no pipeline's memory, imageblock or persistent-threadgroup queries yet
// (RgfxPipelineReflection carries none): 0, logged.

- (NSUInteger)allocatedSize
{
	RGFX_UNSUPPORTED("a pipeline's allocated size (not in the host's reflection)");
	return 0;
}

- (NSUInteger)imageblockMemoryLengthForDimensions:(MTLSize)imageblockDimensions
{
	RGFX_UNSUPPORTED("imageblock memory queries (not in the host's reflection)");
	return 0;
}

- (NSUInteger)recommendedPersistentThreadgroupsPerGridForThreadsPerThreadgroup:
    (MTLSize)threadsPerThreadgroup
{
	RGFX_UNSUPPORTED("persistent threadgroup queries (not in the host's reflection)");
	return 0;
}

// --- function pointers ---------------------------------------------------------------
// Where the host makes them (RgfxSerializer::carriesFunctionPointers): the handle of a function
// the kernel links and a table of the pipeline's, the host's (RgfxFunctionTable.h). Else nil,
// logged, as before the host made any.

// One handle object for a function each time it is asked, as the AMD Radeon Pro 5300M answers
// (measured: macOS 26.7.1, oracle_fnptr); nil, also kept, for a function the kernel does not link.
- (id<MTLFunctionHandle>)functionHandleWithFunction:(id<MTLFunction>)function
{
	if (_serializer == nullptr || !_serializer->carriesFunctionPointers()) {
		RGFX_UNSUPPORTED("function pointers");
		return nil;
	}
	if (function == nil)
		return nil;
	std::lock_guard<std::mutex> hold(_handlesLock);
	if (_handles == nil)
		_handles = [NSMapTable mapTableWithKeyOptions:NSPointerFunctionsStrongMemory |
		                                              NSPointerFunctionsObjectPointerPersonality
		                                 valueOptions:NSPointerFunctionsStrongMemory];
	id known = [_handles objectForKey:function];
	if (known != nil)
		return known == [NSNull null] ? nil : known;
	RgfxFunctionHandle *handle = [RgfxFunctionHandle handleOfFunction:function
	                                                         pipeline:_objId
	                                                           device:self.device
	                                                       serializer:_serializer];
	[_handles setObject:handle != nil ? (id)handle : [NSNull null] forKey:function];
	return handle;
}

// The AMD Radeon Pro 5300M's pipeline raises an unrecognized selector here, answering
// respondsToSelector: YES (measured: macOS 26.7.1, oracle_fnptr), and so does this.
- (id<MTLFunctionHandle>)functionHandleWithName:(NSString *)name
{
	(void)name;
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	if (_serializer == nullptr || !_serializer->carriesFunctionPointers()) {
		RGFX_UNSUPPORTED("function pointers");
		return nil;
	}
	return [RgfxVisibleFunctionTable tableOfPipeline:_objId
	                                           count:descriptor.functionCount
	                                          device:self.device
	                                      serializer:_serializer];
}

// --- unsupported: binary functions --------------------------------------------------

- (id<MTLFunctionHandle>)functionHandleWithBinaryFunction:(id<MTL4BinaryFunction>)function
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithAdditionalBinaryFunctions:(NSArray<id<MTLFunction>> *)functions
                                                   error:(__autoreleasing NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithBinaryFunctions:(NSArray<id<MTL4BinaryFunction>> *)additionalBinaryFunctions
                                         error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

- (id<MTLIntersectionFunctionTable>)newIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

@end
