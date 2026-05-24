"""Build a hand-crafted static ONNX for task389 (color-5 marker swap).

Rule (verified 266/266): wherever input has color 5, output gets color X (the
other non-zero color); wherever input has color X, output gets color 0.

Static-ONNX recipe (v3 — permutation via Gather, no explicit zeros tensor):
  counts  = ReduceSum(input, axes=[2,3])
  X_idx   = ArgMax(counts * mask_not_0_5, axis=1)
  # Build channel-permutation vector p of shape (10,):
  #   p[0]     = X_idx   (so output ch 0 = input ch X)
  #   p[X_idx] = 5       (so output ch X = input ch 5)
  #   p[other] = 0       (output ch other = input ch 0, which is all-zero
  #                       for this task because input never has color 0)
  p = ScatterND(ScatterND(zeros_10, [[0]], X_idx), [[X_idx]], [5])
  output = Gather(input, p, axis=1)        # (1, 10, 30, 30) — final result

This avoids the 36KB "zeros" intermediate of v2: we use Gather to materialise
the output in one step from a tiny (10,) index tensor.
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
    mask_not_0_5 = numpy_helper.from_array(
        np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.float32), "mask_not_0_5")
    axes_23 = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes_23")
    shape_1_1 = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "shape_1_1")
    # initial permutation tensor = zeros(10) int64
    p_zeros = numpy_helper.from_array(np.zeros((10,), dtype=np.int64), "p_zeros")
    # First-scatter index: write at position 0
    idx_at_0 = numpy_helper.from_array(np.array([[0]], dtype=np.int64), "idx_at_0")
    # Second-scatter update value: scalar 5
    val_5 = numpy_helper.from_array(np.array([5], dtype=np.int64), "val_5")

    nodes = []
    nodes.append(helper.make_node("ReduceSum", ["input", "axes_23"], ["counts"], keepdims=0))
    nodes.append(helper.make_node("Mul", ["counts", "mask_not_0_5"], ["masked"]))
    nodes.append(helper.make_node("ArgMax", ["masked"], ["X_idx"], axis=1, keepdims=0))

    # p[0] = X_idx
    nodes.append(helper.make_node(
        "ScatterND", ["p_zeros", "idx_at_0", "X_idx"], ["p_step1"]))
    # p[X_idx] = 5 — reshape X_idx (shape (1,)) to (1,1) for ScatterND indices
    nodes.append(helper.make_node("Reshape", ["X_idx", "shape_1_1"], ["X_idx_2d"]))
    nodes.append(helper.make_node(
        "ScatterND", ["p_step1", "X_idx_2d", "val_5"], ["p"]))

    # output = Gather(input, p, axis=1)
    nodes.append(helper.make_node("Gather", ["input", "p"], ["output"], axis=1))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task389",
        inputs=[inp], outputs=[out],
        initializer=[mask_not_0_5, axes_23, shape_1_1, p_zeros, idx_at_0, val_5],
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
    path = OUTPUT_DIR / "task389.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
