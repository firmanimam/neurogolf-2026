"""Build a hand-crafted static ONNX for task129 (fill grid with most-frequent color).

Rule (verified 265/265): every output cell takes the most-frequent color in the
input grid. Cells outside the actual grid extent stay zero.

Recipe (v1 — sel * mask broadcast):
  counts    = ReduceSum(input, axes=[2,3])                   # (1,10)
  winner    = ArgMax(counts, axis=1)                         # (1,)
  mask      = ReduceSum(input, axes=[1], keepdims=1)         # (1,1,30,30)
  sel       = ScatterND(zeros_10, [[winner]], [1.0])          # (10,) one-hot
  sel_3d    = Reshape(sel, (10,1,1))                          # (10,1,1)
  output    = Mul(sel_3d, mask)                               # broadcasts to (1,10,30,30)

Cost: ~66.9K (27K MACs + 39.8K mem + 19 params).

We tried v2 (ScatterND into a (10,30,30) zeros tensor to skip the broadcast Mul):
it added 9000 params (zeros init) and roughly doubled intermediate memory, ending
at ~138K cost. v1 is cheaper. ConstantOfShape would help but is flagged as risky
on this grader, so we keep the small initializer route.
"""

from __future__ import annotations

import pathlib
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output" / "handbuild_onnx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_model() -> onnx.ModelProto:
    axes_23 = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes_23")
    axes_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_1")
    sel_zeros = numpy_helper.from_array(np.zeros((10,), dtype=np.float32), "sel_zeros")
    val_1f = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "val_1f")
    shape_1_1 = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "shape_1_1")
    shape_10_1_1 = numpy_helper.from_array(np.array([10, 1, 1], dtype=np.int64), "shape_10_1_1")

    nodes = []
    nodes.append(helper.make_node("ReduceSum", ["input", "axes_23"], ["counts"], keepdims=0))
    nodes.append(helper.make_node("ArgMax", ["counts"], ["winner"], axis=1, keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["input", "axes_1"], ["mask"], keepdims=1))
    nodes.append(helper.make_node("Reshape", ["winner", "shape_1_1"], ["winner_2d"]))
    nodes.append(helper.make_node(
        "ScatterND", ["sel_zeros", "winner_2d", "val_1f"], ["sel"]))
    nodes.append(helper.make_node("Reshape", ["sel", "shape_10_1_1"], ["sel_3d"]))
    nodes.append(helper.make_node("Mul", ["sel_3d", "mask"], ["output"]))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task129",
        inputs=[inp], outputs=[out],
        initializer=[axes_23, axes_1, sel_zeros, val_1f, shape_1_1, shape_10_1_1],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    model = build_model()
    path = OUTPUT_DIR / "task129.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
