"""Build a hand-crafted static ONNX for task229 (keep anchor, others -> 5).

Rule: anchor = non-zero color with highest spatial count (ties -> lowest idx).
Output: preserve color 0 and anchor cells; all other non-zero cells become 5.

v3 recipe — Einsum to fuse mask-mul + reduce; minimal intermediates.

  counts        = ReduceSum(input, axes_23, keepdims=0)            # (1,10)
  masked        = counts * mask_not_0                              # zero ch0
  anchor_idx    = ArgMax(masked, axis=1, keepdims=0)               # (1,)
  anchor_oh     = ScatterND(zeros_10, anchor_idx_2d, val_1f)       # (10,)
  keep_mask     = Max(anchor_oh, ch0_oh)                           # (10,)
  kept          = input * Reshape(keep_mask, (1,10,1,1))           # (1,10,30,30)
  # merged = sum over c in 1..9 except anchor: input[c]
  merge_mask    = not0_oh - anchor_oh                              # (10,)
  merged_3d     = Einsum("nchw,c->nhw", input, merge_mask)         # (1,30,30)
  output        = ScatterND(kept, [[0,5]], merged_3d)              # (1,10,30,30)
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
    inits = [
        numpy_helper.from_array(
            np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), "mask_not_0"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes_23"),
        numpy_helper.from_array(np.zeros((10,), dtype=np.float32), "zeros_10_f32"),
        numpy_helper.from_array(
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), "ch0_oh"),
        numpy_helper.from_array(
            np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), "not0_oh"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "val_1f"),
        numpy_helper.from_array(np.array([1, 10, 1, 1], dtype=np.int64), "shape_1_10_1_1"),
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "shape_1_1"),
        numpy_helper.from_array(np.array([[0, 5]], dtype=np.int64), "idx_0_5"),
    ]

    nodes = []
    nodes.append(helper.make_node("ReduceSum", ["input", "axes_23"], ["counts"], keepdims=0))
    nodes.append(helper.make_node("Mul", ["counts", "mask_not_0"], ["masked"]))
    nodes.append(helper.make_node("ArgMax", ["masked"], ["anchor_idx"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Reshape", ["anchor_idx", "shape_1_1"], ["anchor_idx_2d"]))
    nodes.append(helper.make_node(
        "ScatterND", ["zeros_10_f32", "anchor_idx_2d", "val_1f"], ["anchor_oh"]))

    # keep_mask = ch0_oh + anchor_oh (using Max so anchor==0 doesn't double)
    nodes.append(helper.make_node("Max", ["anchor_oh", "ch0_oh"], ["keep_mask"]))
    nodes.append(helper.make_node("Reshape", ["keep_mask", "shape_1_10_1_1"], ["keep_mask_4d"]))
    nodes.append(helper.make_node("Mul", ["input", "keep_mask_4d"], ["kept"]))

    # merge_mask = not0_oh - anchor_oh (1 at c in 1..9, c!=anchor)
    nodes.append(helper.make_node("Sub", ["not0_oh", "anchor_oh"], ["merge_mask"]))
    # merged_3d (1,30,30) = sum over c of input[n,c,h,w] * merge_mask[c]
    nodes.append(helper.make_node(
        "Einsum", ["input", "merge_mask"], ["merged_3d"], equation="nchw,c->nhw"))

    nodes.append(helper.make_node(
        "ScatterND", ["kept", "idx_0_5", "merged_3d"], ["output"]))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task229",
        inputs=[inp], outputs=[out],
        initializer=inits,
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
    path = OUTPUT_DIR / "task229.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
