# seiza-model

Learned restoration of linear astrophotography images. A model is a graph and
a card; this crate does everything around the graph, the same on every
backend: the sky, the noise input, tiling, blending and the bands of noise
kept. A backend only runs tiles, through ONNX Runtime.

A noise model (`NoiseModel`) reads the image and a noise map made from it. A
sharpening model (`SharpenModel`) also reads the image's PSF field, measured
with `seiza-stars`, and its point sources, and is asked for two numbers: the
shares of their width that stars, and everything else, are brought to. Where
its card has a scaled pass, an image whose stars are far wider than the model
was trained on is also sharpened binned, and the two runs are put together by
scale.

A build links nothing native. ONNX Runtime is found at run time, in this
order: a path the host gives, `SEIZA_ONNXRUNTIME`, beside the executable,
`../lib/seiza` from it, the shared data directory, then the system.

`ModelStore::download` installs a model Seiza publishes, by name or the latest
for a task (`Published::Latest(Task::Denoise)`), fetched and verified through
[`seiza-download`](https://crates.io/crates/seiza-download). It is the
default `download` feature; a host that ships its own models can build with
`default-features = false` and install them with `ModelStore::install`.

See [the runtime plan](../docs/design/ml-runtime.md).
