//! The rt platform's libraries ([`RtLibrary`]) and functions ([`RtFunction`]): the translator's,
//! which rt's pipeline creators take.
//!
//! - **A library** is the translator's of a metallib (`newLibraryWithData:error:`), parsed from
//!   its container's tags alone: no function is decoded until a pipeline compiles it. Other bytes,
//!   raw AIR and a bitcode wrapper among them, are refused as Metal refuses them, with an
//!   `NSError`, counted (`rt-refused-translation`), as is a metallib the translator cannot read.
//!   The platform's own shaders (`newLibraryWithSource:options:error:`) are the metallib the build
//!   makes of `shaders.metal` ([`crate::gpu::shaders::METALLIB`]), read once per device, and each
//!   ask answers the same library.
//! - **A function** is a function of its library's table as it was made: with the constant values
//!   it was specialized with, by index (`newFunctionWithName:constantValues:error:`), and the name
//!   and options it was made under (`newFunctionWithDescriptor:error:`), as the translator's
//!   `LinkedFunction` holds them. A pipeline compiles it, and checks its constant values then:
//!   Metal checks them as it makes the function, so a value of the wrong type, or a required
//!   constant given none, answers nil at Metal's function and at rt's pipeline. A name the table
//!   lacks is nil, as Metal answers. A constant of a type no function constant takes, and options
//!   Metal names none of, are Metal's refusal: nil, counted (`rt-refused-metal`).
//!   `MTLFunctionOptionCompileToBinary` is the function's `precompiled`, which rt checks of a
//!   binary function; the other options change nothing rt compiles.

use std::sync::OnceLock;

use metal2vulkan::error::{DescriptorError, Error as TranslatorError};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, FunctionOptions};
use metal2vulkan::pipeline::LinkedFunction;
use metal2vulkan_runtime as rt;

use crate::core::platform::Handle;
use crate::core::platform::resource::{FunctionConstant, Library};
use crate::core::platform::types::NsErrorText;
use crate::core::raise::{OnFault, RaiseResult};

use super::texture::refused_metal;
use super::unbuilt::{rt_refused_create, rt_refused_error, translation};
use super::{RtObj, RtPlatform};

/// `newLibraryWithData:error:`.
const LIBRARY: &str = "newLibraryWithData:error:";

/// `newLibraryWithSource:options:error:`, which core asks the platform's shaders by.
const SHADERS: &str = "newLibraryWithSource:options:error:";

/// `id<MTLLibrary>` on metal2vulkan-runtime: the translator's library.
pub struct RtLibrary {
    rt: rt::Library,
}

impl RtLibrary {
    /// The translator's library.
    pub fn rt(&self) -> &rt::Library {
        &self.rt
    }
}

/// `id<MTLFunction>` on metal2vulkan-runtime: a function of a library, as it was made (the module's
/// docs).
pub struct RtFunction {
    made: LinkedFunction,
}

impl RtFunction {
    /// The function as it was made, as a pipeline links it.
    pub fn linked(&self) -> &LinkedFunction {
        &self.made
    }

    /// The function as a pipeline's stage: its library function, its constant values, and
    /// whether it was made specialized.
    pub fn stage(&self) -> rt::FunctionDescriptor {
        let mut stage = rt::FunctionDescriptor::new(self.made.function.clone());
        stage.constants = self.made.constants.clone();
        stage.specialized = self.made.specialized;
        stage
    }

    /// The name the function answers: its specialized name where it was made with one, else its
    /// library function's own.
    pub fn name(&self) -> &str {
        self.made.name()
    }
}

/// The library of `bytes` (`newLibraryWithData:error:`, the module's docs): `Err` is the
/// `NSError` of bytes the translator reads no metallib of, counted.
pub fn new_library(bytes: &[u8]) -> Result<RtObj<RtLibrary>, NsErrorText> {
    match rt::Library::from_metallib(bytes) {
        Ok(rt) => Ok(RtObj::new(RtLibrary { rt })),
        Err(error) => Err(rt_refused_error(LIBRARY, &translation(error))),
    }
}

/// The platform's shaders as a library, read the first time `shaders` is asked and kept there:
/// each ask answers a new reference to it. `Err` is the `NSError` of the translator's refusal of
/// the metallib, counted at each ask.
pub fn shader_library(
    shaders: &OnceLock<RtObj<RtLibrary>>,
) -> Result<RtObj<RtLibrary>, NsErrorText> {
    if let Some(library) = shaders.get() {
        return Ok(library.clone_ref());
    }
    match rt::Library::from_metallib(crate::gpu::shaders::METALLIB) {
        Ok(rt) => Ok(shaders
            .get_or_init(|| RtObj::new(RtLibrary { rt }))
            .clone_ref()),
        Err(error) => Err(rt_refused_error(SHADERS, &translation(error))),
    }
}

impl RtObj<RtLibrary> {
    /// The function named `name` made as `made` makes it of the library function, by the creator
    /// `sel`: nil for a name the table lacks, as Metal answers, and for one the translator refuses,
    /// counted.
    fn made(
        &self,
        sel: &'static str,
        name: &str,
        made: impl FnOnce(LinkedFunction) -> LinkedFunction,
    ) -> Option<RtObj<RtFunction>> {
        match self.rt.function(name) {
            Ok(function) => Some(RtObj::new(RtFunction {
                made: made(LinkedFunction::new(function)),
            })),
            Err(TranslatorError::InvalidDescriptor(DescriptorError::EntryNotFound { .. })) => None,
            Err(error) => {
                rt_refused_create(sel, &translation(error));
                None
            }
        }
    }
}

impl Library<RtPlatform> for RtObj<RtLibrary> {
    /// Every function of the table, in its order.
    fn function_names(&self) -> Vec<String> {
        (self.rt.functions())
            .map(|info| info.name.to_owned())
            .collect()
    }

    /// The function unspecialized, as `newFunctionWithName:` makes it.
    fn new_function(
        &self,
        on: OnFault,
        function_name: &str,
    ) -> RaiseResult<Option<RtObj<RtFunction>>> {
        let recorded_call = crate::recorder::rec_call!("newFunctionWithName:", self, function_name);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtFunction>>> {
            let _ = on;
            Ok(self.made("newFunctionWithName:", function_name, |f| f))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The function specialized with `constants` (the module's docs).
    fn new_function_with_constants(
        &self,
        on: OnFault,
        function_name: &str,
        constants: &[FunctionConstant],
    ) -> RaiseResult<Option<RtObj<RtFunction>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newFunctionWithName:constantValues:error:",
            self,
            function_name,
            constants
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtFunction>>> {
            let _ = on;
            let sel = "newFunctionWithName:constantValues:error:";
            let Some(values) = constant_values(sel, constants) else {
                return Ok(None);
            };
            Ok(self.made(sel, function_name, |f| f.with_constants(values)))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The function specialized, with `constants` where there are any, under `specialized_name`
    /// where there is one, and compiled to a binary where `options` says so (the module's docs).
    fn new_function_with_options(
        &self,
        on: OnFault,
        function_name: &str,
        constants: &[FunctionConstant],
        options: u64,
        specialized_name: Option<&str>,
    ) -> RaiseResult<Option<RtObj<RtFunction>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newFunctionWithDescriptor:error:",
            self,
            function_name,
            constants,
            options,
            specialized_name
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtFunction>>> {
            let _ = on;
            let sel = "newFunctionWithDescriptor:error:";
            let options = u32::try_from(options).ok().map(FunctionOptions::try_from);
            let Some(Ok(options)) = options else {
                return Ok(refused_metal(sel, &"function options Metal names none of"));
            };
            let Some(values) = constant_values(sel, constants) else {
                return Ok(None);
            };
            Ok(self.made(sel, function_name, |f| {
                let f = f
                    .with_constants(values)
                    .with_precompiled(options.contains(FunctionOptions::COMPILE_TO_BINARY));
                match specialized_name {
                    Some(name) => f.with_specialized_name(name),
                    None => f.with_specialized(true),
                }
            }))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }
}

/// The translator's values of `constants`, each by its index: `None` where one is of a type no
/// function constant takes, Metal's refusal, counted.
fn constant_values(
    sel: &'static str,
    constants: &[FunctionConstant],
) -> Option<FunctionConstantValues> {
    let mut values = FunctionConstantValues::defaults();
    for c in constants {
        let (Ok(ty), Some(bytes)) = (DataType::try_from(c.data_type), c.bytes()) else {
            let why = format!(
                "constant {} of MTLDataType {}, which no function constant takes",
                c.index, c.data_type
            );
            return refused_metal(sel, &why);
        };
        values = values.with_index(c.index, FunctionConstantValue::new(ty, bytes.to_vec()));
    }
    Some(values)
}
