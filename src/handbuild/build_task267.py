"""Build a hand-crafted static ONNX for task267 (shape recolored by single marker).

Rule (verified 264/264): input contains two non-zero colors —
  X = the more frequent (shape body)
  Y = the less frequent (single-cell marker)
Output: positions where input == X get color Y; everything else 0.

In one-hot terms this is a channel-mix matrix M (10,10) with
  output[c] = sum_i M_T[c,i] * input[i]
where M_T[c,i] is 1 iff input channel i should contribute to output channel c:
  M_T[0, 0] = 1  (background stays as 0)
  M_T[0, Y] = 1  (the marker pixel becomes 0)
  M_T[Y, X] = 1  (shape body becomes Y)
otherwise 0.

We dynamically build M_T (10,10) with 2 ScatterND inserts onto a pre-baked
matrix that already has M_T[0,0]=1, then apply via a single MatMul on a
(1,10,900) flattened view of the input.

Cost rationale: a single MatMul((10,10)@(1,10,900)) = 90,000 MACs and the
output (1,10,900) is exactly 36,000 bytes — same as (1,10,30,30) — so the
final Reshape adds no copy. There is only ONE (1,10,30,30)-sized intermediate
in the whole graph, vs three in the broadcast-multiply approach.
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
    mask_not_0 = numpy_helper.from_array(
        np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), "mask_not_0")
    axes_23 = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes_23")
    shape_1_1 = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "shape_1_1")
    shape_1_10_900 = numpy_helper.from_array(
        np.array([1, 10, 900], dtype=np.int64), "shape_1_10_900")
    shape_1_10_30_30 = numpy_helper.from_array(
        np.array([1, 10, 30, 30], dtype=np.int64), "shape_1_10_30_30")
    zeros_10_f = numpy_helper.from_array(np.zeros((10,), dtype=np.float32), "zeros_10_f")
    ones_10_f = numpy_helper.from_array(np.ones((10,), dtype=np.float32), "ones_10_f")
    one_f = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one_f")
    one_f_2 = numpy_helper.from_array(np.array([1.0, 1.0], dtype=np.float32), "one_f_2")
    # M_T0: 10x10 zeros with M_T0[0,0]=1 (background -> background)
    M_T0_arr = np.zeros((10, 10), dtype=np.float32)
    M_T0_arr[0, 0] = 1.0
    M_T0 = numpy_helper.from_array(M_T0_arr, "M_T0")
    # Constant (1,1) zero for building [0, Y_idx] index
    zero_1_1 = numpy_helper.from_array(np.zeros((1, 1), dtype=np.int64), "zero_1_1")

    nodes = []
    # --- find X_idx and Y_idx ----------------------------------------------
    nodes.append(helper.make_node("ReduceSum", ["input", "axes_23"], ["counts"], keepdims=0))
    nodes.append(helper.make_node("Mul", ["counts", "mask_not_0"], ["masked_nz"]))
    nodes.append(helper.make_node("ArgMax", ["masked_nz"], ["X_idx"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Reshape", ["X_idx", "shape_1_1"], ["X_idx_11"]))
    nodes.append(helper.make_node(
        "ScatterND", ["zeros_10_f", "X_idx_11", "one_f"], ["X_oh"]))
    nodes.append(helper.make_node("Sub", ["ones_10_f", "X_oh"], ["inv_X_oh"]))
    nodes.append(helper.make_node("Mul", ["masked_nz", "inv_X_oh"], ["masked_no_X"]))
    nodes.append(helper.make_node("ArgMax", ["masked_no_X"], ["Y_idx"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Reshape", ["Y_idx", "shape_1_1"], ["Y_idx_11"]))

    # --- build the (10,10) mix matrix M_T ---------------------------------
    # We need to set:
    #   M_T[0,    Y] = 1   →  index pair (0, Y_idx)
    #   M_T[Y,    X] = 1   →  index pair (Y_idx, X_idx)
    # Each "index pair" must be shape (1,2) for a single ScatterND update.
    # Build them via Concat along axis=1 of two (1,1) int64 tensors.

    # idx_0Y = concat([[0]], [[Y]]) → (1,2)
    nodes.append(helper.make_node(
        "Concat", ["zero_1_1", "Y_idx_11"], ["idx_0Y"], axis=1))
    # idx_YX = concat([[Y]], [[X]]) → (1,2)
    nodes.append(helper.make_node(
        "Concat", ["Y_idx_11", "X_idx_11"], ["idx_YX"], axis=1))
    # Stack both indices into (2,2) so we can do one ScatterND with both updates.
    nodes.append(helper.make_node(
        "Concat", ["idx_0Y", "idx_YX"], ["idx_both"], axis=0))      # (2,2)
    nodes.append(helper.make_node(
        "ScatterND", ["M_T0", "idx_both", "one_f_2"], ["M_T"]))     # (10,10)

    # --- apply mix: output = M_T @ flat_input -----------------------------
    nodes.append(helper.make_node("Reshape", ["input", "shape_1_10_900"], ["flat_in"]))
    nodes.append(helper.make_node("MatMul", ["M_T", "flat_in"], ["flat_out"]))  # (1,10,900)
    nodes.append(helper.make_node("Reshape", ["flat_out", "shape_1_10_30_30"], ["output"]))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task267",
        inputs=[inp], outputs=[out],
        initializer=[
            mask_not_0, axes_23, shape_1_1,
            shape_1_10_900, shape_1_10_30_30,
            zeros_10_f, ones_10_f, one_f, one_f_2,
            M_T0, zero_1_1,
        ],
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
    path = OUTPUT_DIR / "task267.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
