"""Write the small graphs that hold the runtime to what a card says of a model's
shape. Each is a 1 x 1 convolution, so a pixel's answer can be worked out by
hand, and the tests in `tests/runtime.rs` do that. It needs only `onnx`:

    python seiza-model/tests/fixtures/make_shapes.py

The bundles from `three-planes` on are wrong on purpose: their cards and graphs
disagree, and the runtime has to say so when it opens them.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).parent / "shapes"
TILE = 64
# The weights a graph gives the image and the noise plane. They differ, so a
# plane in the wrong place changes the answer.
IMAGE, NOISE = 0.25, 0.5
FREE = ["height", "width"]


def tensor(name: str, planes: int, sides: list, element: int = TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, element, [1, planes, *sides])


def conv(source: str, weights: list[list[float]], target: str, dtype=np.float32) -> tuple:
    """A convolution from `source` to `target`, one row of `weights` per plane out."""
    kernel = np.array(weights, dtype=dtype)[:, :, None, None]
    initializer = numpy_helper.from_array(kernel, name=f"{target}-weights")
    return helper.make_node("Conv", [source, initializer.name], [target]), initializer


def graph(nodes: list, initializers: list, inputs: list, outputs: list) -> bytes:
    made = helper.make_graph(nodes, "shape", inputs, outputs, initializers)
    model = helper.make_model(made, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def bundle(name: str, model: bytes, inputs: list, *, precision: str = "f32", dimensions=FREE, tile: int = TILE,
           output: str = "residual") -> None:
    root = HERE / name
    root.mkdir(parents=True)
    file = f"model-{precision}.onnx"
    (root / file).write_bytes(model)
    card = {
        "format": 1,
        "name": name,
        "task": "denoise",
        "graphs": [{"precision": precision, "file": file, "sha256": hashlib.sha256(model).hexdigest()}],
        "inputs": inputs,
        "output": output,
        "dimensions": list(dimensions),
        "reading": {"noise_map": "constant", "noise_plane": "asinh", "sky": "global", "sky_block": 32,
                    "flat_block": 64, "image_map_scale": 1.13},
        "tiling": {"tile": tile, "margin": 10, "ramp": 8, "multiple": 2},
        "parity": ["cpu"],
    }
    (root / "card.json").write_text(json.dumps(card, indent=2) + "\n")


def planes(*names: str, tensor_name: str = "planes") -> dict:
    return {"tensor": tensor_name, "planes": list(names)}


def write_fits() -> None:
    """A corner of the fixture image as FITS, for the packages' clean-machine tests."""
    width, height = 300, 230
    pixels = np.fromfile(HERE.parent / "image.f32", dtype="<f4").reshape(height, width)[:96, :128]
    cards = ["SIMPLE  =                    T", "BITPIX  =                  -32", "NAXIS   =                    2",
             f"NAXIS1  = {pixels.shape[1]:20d}", f"NAXIS2  = {pixels.shape[0]:20d}", "END"]
    header = "".join(card.ljust(80) for card in cards).ljust(2880).encode("ascii")
    data = pixels.astype(">f4").tobytes()
    (HERE.parent / "tiny-input.fits").write_bytes(header + data + bytes(-len(data) % 2880))


def main() -> None:
    write_fits()
    shutil.rmtree(HERE, ignore_errors=True)
    two = [planes("image", "noise")]
    residual = tensor("residual", 1, FREE)

    node, weights = conv("planes", [[IMAGE, NOISE]], "residual")
    plain = graph([node], [weights], [tensor("planes", 2, FREE)], [residual])
    bundle("two-planes", plain, two)

    node, weights = conv("picture", [[IMAGE]], "residual")
    bundle("one-plane", graph([node], [weights], [tensor("picture", 1, FREE)], [residual]),
           [planes("image", tensor_name="picture")])

    node, weights = conv("planes", [[NOISE, IMAGE]], "residual")
    bundle("reversed", graph([node], [weights], [tensor("planes", 2, FREE)], [residual]),
           [planes("noise", "image")])

    joined = helper.make_node("Concat", ["picture", "level"], ["planes"], axis=1)
    node, weights = conv("planes", [[IMAGE, NOISE]], "residual")
    bundle("two-tensors", graph([joined, node], [weights], [tensor("picture", 1, FREE), tensor("level", 1, FREE)],
                                [residual]),
           [planes("image", tensor_name="picture"), planes("noise", tensor_name="level")])

    node, weights = conv("planes", [[IMAGE, NOISE]], "residual")
    bundle("fixed", graph([node], [weights], [tensor("planes", 2, [TILE, TILE])],
                          [tensor("residual", 1, [TILE, TILE])]), two, dimensions=[])

    # f16 inside, with f32 at both ends, as a converter that keeps a graph's
    # types at its edges writes.
    narrow = helper.make_node("Cast", ["planes"], ["narrow"], to=TensorProto.FLOAT16)
    node, weights = conv("narrow", [[IMAGE, NOISE]], "wide", np.float16)
    widen = helper.make_node("Cast", ["wide"], ["residual"], to=TensorProto.FLOAT)
    bundle("half-inside", graph([narrow, node, widen], [weights], [tensor("planes", 2, FREE)], [residual]), two,
           precision="f16")

    # f16 at both ends, as the bundle exporter writes, and at one end only.
    half = TensorProto.FLOAT16
    node, weights = conv("planes", [[IMAGE, NOISE]], "residual", np.float16)
    bundle("half-tensors", graph([node], [weights], [tensor("planes", 2, FREE, half)],
                                 [tensor("residual", 1, FREE, half)]), two, precision="f16")

    node, weights = conv("planes", [[IMAGE, NOISE]], "wide", np.float16)
    widen = helper.make_node("Cast", ["wide"], ["residual"], to=TensorProto.FLOAT)
    bundle("half-in", graph([node, widen], [weights], [tensor("planes", 2, FREE, half)], [residual]), two,
           precision="f16")

    # Wrong on purpose from here.
    node, weights = conv("planes", [[IMAGE, NOISE, 0.0]], "residual")
    bundle("three-planes", graph([node], [weights], [tensor("planes", 3, FREE)], [residual]), two)

    node, weights = conv("planes", [[IMAGE, NOISE]] * 3, "residual")
    bundle("three-out", graph([node], [weights], [tensor("planes", 2, FREE)], [tensor("residual", 3, FREE)]), two)

    bundle("other-input", plain, [planes("image", "noise", tensor_name="pixels")])

    node, weights = conv("picture", [[IMAGE]], "residual")
    bundle("missing-input", graph([node], [weights], [tensor("picture", 1, FREE)], [residual]),
           [planes("image", tensor_name="picture"), planes("noise", tensor_name="level")])

    bundle("other-output", plain, two, output="clean")

    node, weights = conv("planes", [[IMAGE, NOISE]], "residual")
    add = helper.make_node("Add", ["residual", "extra"], ["total"])
    bundle("extra-input", graph([node, add], [weights], [tensor("planes", 2, FREE), tensor("extra", 1, FREE)],
                                [tensor("total", 1, FREE)]), two, output="total")

    node, weights = conv("planes", [[IMAGE, NOISE]], "residual")
    bundle("fixed-small", graph([node], [weights], [tensor("planes", 2, [32, 32])],
                                [tensor("residual", 1, [32, 32])]), two, dimensions=[])


if __name__ == "__main__":
    main()
