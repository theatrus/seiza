//! The one place that touches ONNX Runtime's session API, since `ort` is a
//! release candidate and it may change.

use crate::backend::{Backend, CoreMlUnits, Runtime};
use crate::bundle::ModelCard;
use crate::error::ModelError;
use crate::tiles::TileRunner;
use half::f16;
use ort::ep::{self, ExecutionProviderDispatch, coreml};
use ort::session::{Session, SessionInputValue};
use ort::value::{Outlet, TensorElementType, TensorRef, ValueType};
use std::borrow::Cow;
use std::path::Path;

/// What a tensor holds. A graph says this for itself, and it need not follow
/// the graph's precision: an f16 graph may read and write f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Element {
    F32,
    F16,
}

/// One of the graph's input tensors.
struct Input {
    name: String,
    planes: usize,
    element: Element,
    half: Vec<f16>,
}

/// A session fixed to one tile's size.
pub(crate) struct OrtRunner {
    session: Session,
    inputs: Vec<Input>,
    output: String,
    output_element: Element,
}

/// The provider for `backend`, or `None` for ONNX Runtime's own CPU provider.
fn provider(
    backend: Backend,
    cache: Option<&Path>,
    device_id: Option<i32>,
) -> Option<ExecutionProviderDispatch> {
    let units = |units: CoreMlUnits| {
        let units = match units {
            CoreMlUnits::NeuralEngine => coreml::ComputeUnits::CPUAndNeuralEngine,
            CoreMlUnits::Gpu => coreml::ComputeUnits::CPUAndGPU,
            CoreMlUnits::Cpu => coreml::ComputeUnits::CPUOnly,
        };
        let provider = ep::CoreML::default()
            .with_model_format(coreml::ModelFormat::MLProgram)
            .with_compute_units(units);
        match cache {
            Some(path) => provider.with_model_cache_dir(path.display().to_string()),
            None => provider,
        }
    };
    match backend {
        Backend::CoreMl(chosen) => Some(units(chosen).build()),
        Backend::TensorRt => Some(ep::TensorRT::default().build()),
        Backend::TensorRtRtx => Some(ep::NVRTX::default().build()),
        Backend::Cuda => Some(ep::CUDA::default().build()),
        Backend::DirectMl => {
            let ep = ep::DirectML::default();
            let ep = match device_id {
                Some(index) => ep.with_device_id(index),
                None => ep.with_performance_preference(
                    ep::directml::PerformancePreference::HighPerformance,
                ),
            };
            Some(ep.build())
        }
        Backend::WebGpu => Some(ep::WebGPU::default().build()),
        Backend::Cpu => None,
    }
}

/// What `outlet` holds, once its shape is `planes` planes of `side` × `side`.
fn fitted(
    outlet: &Outlet,
    role: &str,
    planes: usize,
    side: usize,
    model: &str,
) -> Result<Element, ModelError> {
    fitted_to(outlet, role, planes, side, side, model)
}

/// What `outlet` holds, once its shape is `planes` planes of `wide` × `high`.
/// A dimension the graph leaves free passes. A tensor that does not fit is
/// refused here, since at run time it would stop at the first tile at best,
/// and at worst keep part of what the network wrote.
fn fitted_to(
    outlet: &Outlet,
    role: &str,
    planes: usize,
    high: usize,
    wide: usize,
    model: &str,
) -> Result<Element, ModelError> {
    let name = outlet.name();
    let refuse = |reason: String| {
        Err(ModelError::ModelContract(format!(
            "{model}'s {role} {name} {reason}"
        )))
    };
    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return refuse("is not a tensor".to_string());
    };
    let element = match ty {
        TensorElementType::Float32 => Element::F32,
        TensorElementType::Float16 => Element::F16,
        other => return refuse(format!("holds {other}, not f32 or f16")),
    };
    let &[batch, channels, height, width] = &shape[..] else {
        return refuse(format!(
            "has {} dimensions, not batch, planes, height and width",
            shape.len()
        ));
    };
    let fits = |dimension: i64, wanted: usize| dimension < 0 || dimension == wanted as i64;
    if !fits(batch, 1) {
        return refuse(format!(
            "takes a batch of {batch}, and tiles run one by one"
        ));
    }
    if !fits(channels, planes) {
        return refuse(format!("holds {channels} planes, not {planes}"));
    }
    if !fits(height, high) || !fits(width, wide) {
        return refuse(format!(
            "is {width} × {height}, and this run gives it {wide} × {high}"
        ));
    }
    Ok(element)
}

/// A tensor as a run fixes it: its name, and `planes` planes of `width` ×
/// `height`.
#[derive(Clone, Debug)]
pub(crate) struct TensorShape {
    pub(crate) name: String,
    pub(crate) planes: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

impl TensorShape {
    fn len(&self) -> usize {
        self.planes * self.height * self.width
    }
}

/// A session over a graph of several input tensors, each of a size of its
/// own, and one output: a sharpening model's encoder, and its body.
pub(crate) struct OrtGraph {
    session: Session,
    inputs: Vec<(TensorShape, Element, Vec<f16>)>,
    output: TensorShape,
    output_element: Element,
}

impl OrtGraph {
    /// Open `graph` with each free dimension in `dimensions` set, and hold
    /// its tensors to `inputs` and `output`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open(
        runtime: &Runtime,
        graph: &Path,
        model: &str,
        backend: Backend,
        dimensions: &[(&str, usize)],
        inputs: &[TensorShape],
        output: &TensorShape,
        cache: Option<&Path>,
        device_id: Option<i32>,
    ) -> Result<Self, ModelError> {
        let mut builder = Session::builder(&runtime.environment)?;
        for (dimension, size) in dimensions {
            builder = builder.with_dimension_override(*dimension, *size as i64)?;
        }
        if backend == Backend::DirectMl {
            builder = builder
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?;
        }
        if let Some(provider) = provider(backend, cache, device_id) {
            // As for a noise model: a graph that does not fit its backend
            // must fail, not run elsewhere.
            builder = builder
                .with_config_entry("session.disable_cpu_ep_fallback", "1")?
                .with_execution_providers([provider.error_on_failure()])?;
        }
        let session = builder.commit_from_file(graph)?;
        let refuse = |reason: String| Err(ModelError::ModelContract(format!("{model}'s {reason}")));
        if let Some(outlet) = session
            .inputs()
            .iter()
            .find(|outlet| !inputs.iter().any(|input| input.name == outlet.name()))
        {
            return refuse(format!(
                "graph reads {}, which its card does not name",
                outlet.name()
            ));
        }
        let mut held = Vec::with_capacity(inputs.len());
        for input in inputs {
            let Some(outlet) = session
                .inputs()
                .iter()
                .find(|outlet| outlet.name() == input.name)
            else {
                return refuse(format!(
                    "card names the input {}, which its graph does not read",
                    input.name
                ));
            };
            let element = fitted_to(
                outlet,
                "input",
                input.planes,
                input.height,
                input.width,
                model,
            )?;
            held.push((input.clone(), element, Vec::new()));
        }
        let Some(outlet) = session
            .outputs()
            .iter()
            .find(|outlet| outlet.name() == output.name)
        else {
            return refuse(format!(
                "card names the output {}, which its graph does not write",
                output.name
            ));
        };
        let output_element = fitted_to(
            outlet,
            "output",
            output.planes,
            output.height,
            output.width,
            model,
        )?;
        Ok(Self {
            session,
            inputs: held,
            output: output.clone(),
            output_element,
        })
    }

    /// Fill `out` from `inputs`, one slice for each tensor in the order
    /// they were opened in.
    pub(crate) fn run(&mut self, inputs: &[&[f32]], out: &mut [f32]) -> Result<(), ModelError> {
        let mut tensors: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(self.inputs.len());
        for ((shape, element, half), values) in self.inputs.iter_mut().zip(inputs) {
            if values.len() != shape.len() {
                return Err(ModelError::ModelContract(format!(
                    "{} takes {} values, not {}",
                    shape.name,
                    shape.len(),
                    values.len()
                )));
            }
            let dimensions = [
                1_i64,
                shape.planes as i64,
                shape.height as i64,
                shape.width as i64,
            ];
            let value = match element {
                Element::F32 => {
                    SessionInputValue::from(TensorRef::from_array_view((dimensions, *values))?)
                }
                Element::F16 => {
                    half.clear();
                    half.extend(values.iter().map(|&value| f16::from_f32(value)));
                    SessionInputValue::from(TensorRef::from_array_view((
                        dimensions,
                        half.as_slice(),
                    ))?)
                }
            };
            tensors.push((Cow::Borrowed(shape.name.as_str()), value));
        }
        let outputs = self.session.run(tensors)?;
        let found = &outputs[self.output.name.as_str()];
        match self.output_element {
            Element::F32 => fill(out, found.try_extract_tensor::<f32>()?.1, |value| value),
            Element::F16 => fill(out, found.try_extract_tensor::<f16>()?.1, |value: f16| {
                value.to_f32()
            }),
        }
    }
}

impl OrtRunner {
    /// Open `graph` with every free dimension set to `side`, and hold it to
    /// `card`. A dynamic graph fixed at load serves every tile from one file.
    pub(crate) fn open(
        runtime: &Runtime,
        graph: &Path,
        card: &ModelCard,
        backend: Backend,
        side: usize,
        cache: Option<&Path>,
        device_id: Option<i32>,
    ) -> Result<Self, ModelError> {
        let mut builder = Session::builder(&runtime.environment)?;
        for dimension in &card.dimensions {
            builder = builder.with_dimension_override(dimension, side as i64)?;
        }
        if backend == Backend::DirectMl {
            builder = builder
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?;
        }
        if let Some(provider) = provider(backend, cache, device_id) {
            // A graph that does not fit its backend must fail, not run
            // elsewhere and report the backend asked for.
            builder = builder
                .with_config_entry("session.disable_cpu_ep_fallback", "1")?
                .with_execution_providers([provider.error_on_failure()])?;
        }
        let session = builder.commit_from_file(graph)?;

        let named = |outlets: &[Outlet]| {
            outlets
                .iter()
                .map(|outlet| outlet.name().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        for outlet in session.inputs() {
            if !card
                .inputs
                .iter()
                .any(|input| input.tensor == outlet.name())
            {
                return Err(ModelError::ModelContract(format!(
                    "{}'s graph reads {}, which its card does not name",
                    card.name,
                    outlet.name()
                )));
            }
        }
        let mut inputs = Vec::with_capacity(card.inputs.len());
        for input in &card.inputs {
            let outlet = session
                .inputs()
                .iter()
                .find(|outlet| outlet.name() == input.tensor)
                .ok_or_else(|| {
                    ModelError::ModelContract(format!(
                        "{}'s card names the input {}, and its graph reads {}",
                        card.name,
                        input.tensor,
                        named(session.inputs())
                    ))
                })?;
            inputs.push(Input {
                name: input.tensor.clone(),
                planes: input.planes.len(),
                element: fitted(outlet, "input", input.planes.len(), side, &card.name)?,
                half: Vec::new(),
            });
        }
        let outlet = session
            .outputs()
            .iter()
            .find(|outlet| outlet.name() == card.output)
            .ok_or_else(|| {
                ModelError::ModelContract(format!(
                    "{}'s card names the output {}, and its graph writes {}",
                    card.name,
                    card.output,
                    named(session.outputs())
                ))
            })?;
        let output_element = fitted(outlet, "output", 1, side, &card.name)?;
        let output = card.output.clone();
        Ok(Self {
            session,
            inputs,
            output,
            output_element,
        })
    }

    /// `planes`, one after the other, as the graph's input tensors.
    fn tensors<'a>(
        inputs: &'a mut [Input],
        mut planes: &'a [f32],
        side: usize,
    ) -> Result<Vec<(Cow<'a, str>, SessionInputValue<'a>)>, ModelError> {
        let mut tensors = Vec::with_capacity(inputs.len());
        for input in inputs {
            let (own, rest) = planes.split_at(input.planes * side * side);
            planes = rest;
            let shape = [1_i64, input.planes as i64, side as i64, side as i64];
            let value = match input.element {
                Element::F32 => SessionInputValue::from(TensorRef::from_array_view((shape, own))?),
                Element::F16 => {
                    input.half.clear();
                    input
                        .half
                        .extend(own.iter().map(|&value| f16::from_f32(value)));
                    SessionInputValue::from(TensorRef::from_array_view((
                        shape,
                        input.half.as_slice(),
                    ))?)
                }
            };
            tensors.push((Cow::Borrowed(input.name.as_str()), value));
        }
        Ok(tensors)
    }
}

/// Copy a tile's result, which a graph with free dimensions could still get
/// wrong, into `residual`.
fn fill<T: Copy>(
    residual: &mut [f32],
    values: &[T],
    widen: impl Fn(T) -> f32,
) -> Result<(), ModelError> {
    if values.len() != residual.len() {
        return Err(ModelError::ModelContract(format!(
            "a tile wrote {} values, not {}",
            values.len(),
            residual.len()
        )));
    }
    for (out, &value) in residual.iter_mut().zip(values) {
        *out = widen(value);
    }
    Ok(())
}

impl TileRunner for OrtRunner {
    fn run(&mut self, planes: &[f32], side: usize, residual: &mut [f32]) -> Result<(), ModelError> {
        let tensors = Self::tensors(&mut self.inputs, planes, side)?;
        let outputs = self.session.run(tensors)?;
        let found = &outputs[self.output.as_str()];
        match self.output_element {
            Element::F32 => fill(residual, found.try_extract_tensor::<f32>()?.1, |value| {
                value
            }),
            Element::F16 => fill(
                residual,
                found.try_extract_tensor::<f16>()?.1,
                |value: f16| value.to_f32(),
            ),
        }
    }
}
