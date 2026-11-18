# seiza-model

Learned restoration of linear astrophotography images. A model is a graph and
a card; this crate does everything around the graph, the same on every
backend: the sky, the noise input, tiling, blending and the bands of noise
kept. A backend only runs tiles, through ONNX Runtime.

A noise model (`NoiseModel`) reads the image and a noise map made from it. A
sharpening model (`SharpenModel`) also reads the image's PSF field, measured
with `seiza-stars`, and its point sources, and is asked for two numbers: the
shares of their width that stars, and everything else, are brought to.

A build links nothing native. ONNX Runtime is found at run time, in this
order: a path the host gives, `SEIZA_ONNXRUNTIME`, beside the executable,
`../lib/seiza` from it, the shared data directory, then the system.

See [the runtime plan](../docs/design/ml-runtime.md).
