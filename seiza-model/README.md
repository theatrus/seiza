# seiza-model

Learned restoration of linear astrophotography images. A model is a graph and
a card; this crate does everything around the graph, the same on every
backend: the sky, the noise input, tiling, blending and the bands of noise
kept. A backend only runs tiles, through ONNX Runtime.

A build links nothing native. ONNX Runtime is found at run time, in this
order: a path the host gives, `SEIZA_ONNXRUNTIME`, beside the executable,
`../lib/seiza` from it, the shared data directory, then the system.

See [the runtime plan](../docs/design/ml-runtime.md).
