//! The one place that touches ONNX Runtime's session API, since `ort` is a
//! release candidate and it may change.

use crate::backend::{Backend, CoreMlUnits, Precision, Runtime};
use crate::bundle::ModelCard;
use crate::error::ModelError;
use crate::tiles::TileRunner;
use half::f16;
use ort::ep::{self, ExecutionProviderDispatch, coreml};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

/// A session fixed to one tile's size, and the graph's precision.
pub(crate) struct OrtRunner {
    session: Session,
    channels: usize,
    precision: Precision,
    half: Vec<f16>,
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

impl OrtRunner {
    /// Open `graph` with every free dimension set to `side`. A dynamic graph
    /// fixed at load serves every tile from one file.
    pub(crate) fn open(
        runtime: &Runtime,
        graph: &Path,
        card: &ModelCard,
        backend: Backend,
        precision: Precision,
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
        Ok(Self {
            session,
            channels: card.input_channels,
            precision,
            half: Vec::new(),
        })
    }
}

impl TileRunner for OrtRunner {
    fn run(&mut self, planes: &[f32], side: usize, residual: &mut [f32]) -> Result<(), ModelError> {
        let shape = [1_i64, self.channels as i64, side as i64, side as i64];
        match self.precision {
            Precision::F32 => {
                let input = TensorRef::from_array_view((shape, planes))?;
                let outputs = self.session.run(inputs![input])?;
                let (_, values) = outputs[0].try_extract_tensor::<f32>()?;
                residual.copy_from_slice(values);
            }
            Precision::F16 => {
                self.half.clear();
                self.half
                    .extend(planes.iter().map(|&value| f16::from_f32(value)));
                let input = TensorRef::from_array_view((shape, self.half.as_slice()))?;
                let outputs = self.session.run(inputs![input])?;
                let (_, values) = outputs[0].try_extract_tensor::<f16>()?;
                for (out, value) in residual.iter_mut().zip(values) {
                    *out = value.to_f32();
                }
            }
        }
        Ok(())
    }
}
