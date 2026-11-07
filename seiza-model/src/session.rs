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
fn provider(backend: Backend, cache: Option<&Path>) -> Option<ExecutionProviderDispatch> {
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
        Backend::WebGpu => Some(ep::WebGPU::default().build()),
        Backend::Cpu => None,
    }
}

/// What `outlet` holds, once its shape is `planes` planes of `side` × `side`.
/// A dimension the graph leaves free passes. A tensor that does not fit is
/// refused here, since at run time it would stop at the first tile at best,
/// and at worst keep part of what the network wrote.
fn fitted(
    outlet: &Outlet,
    role: &str,
    planes: usize,
    side: usize,
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
    if !fits(height, side) || !fits(width, side) {
        return refuse(format!(
            "is {width} × {height}, and this run's tile is {side} × {side}"
        ));
    }
    Ok(element)
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
    ) -> Result<Self, ModelError> {
        let mut builder = Session::builder(&runtime.environment)?;
        for dimension in &card.dimensions {
            builder = builder.with_dimension_override(dimension, side as i64)?;
        }
        if let Some(provider) = provider(backend, cache) {
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
