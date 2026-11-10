# seiza-model

Learned restoration of linear astrophotography images. A model is a graph and
a card; this crate does everything around the graph, the same on every
backend: the sky, the noise input, tiling, blending and the bands of noise
kept. A backend only runs tiles, through ONNX Runtime.

A build links nothing native. ONNX Runtime is found at run time, in this
order: a path the host gives, `SEIZA_ONNXRUNTIME`, beside the executable,
`../lib/seiza` from it, the shared data directory, then the system.

`ModelStore::download` installs a model Seiza publishes, fetched and verified
through [`seiza-download`](https://crates.io/crates/seiza-download). It is the
default `download` feature; a host that ships its own models can build with
`default-features = false` and install them with `ModelStore::install`.

See [the runtime plan](../docs/design/ml-runtime.md).
