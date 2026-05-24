"""Build a hand-crafted static ONNX for task282 (3x3 ring stamp around color-5).

Rule (verified 265/265): For each cell with color 5 in input, stamp a 3x3 ring
centered on it — corners=5, edges=1, center=0 (overwrites marker). Markers
near borders are clipped at the 9x9 grid edge.

Static-ONNX recipe v5 (Slice→Pad masking instead of broadcast Mul):
  m5         = Slice(input, axis=1, [5:6])               # (1,1,30,30)
  in_grid    = ReduceSum(input, axes=[1], keepdims=1)    # (1,1,30,30)
  ch1_raw    = Conv(m5, edges_kernel,   pad=1)           # (1,1,30,30)
  ch5_raw    = Conv(m5, corners_kernel, pad=1)           # (1,1,30,30)
  # Clip conv outputs to actual 9x9 grid (so a marker at (8,8) does not bleed
  # corner to (9,9) outside the data region). Done by Slice axes=[2,3] [0:9]
  # then Pad back to 30x30 with zeros.
  ch1_9     = Slice(ch1_raw, axes=[2,3], [0:9, 0:9])     # (1,1,9,9)
  ch5_9     = Slice(ch5_raw, axes=[2,3], [0:9, 0:9])
  ch1       = Pad(ch1_9, [0,0,0,0, 0,0,21,21])           # (1,1,30,30)
  ch5       = Pad(ch5_9, [0,0,0,0, 0,0,21,21])
  ch0       = in_grid - ch1 - ch5
  zero      = Sub(m5, m5)
  output    = Concat([ch0, ch1, zero, zero, zero, ch5, zero, zero, zero, zero], axis=1)
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
    corners = np.array([[[[1, 0, 1],
                          [0, 0, 0],
                          [1, 0, 1]]]], dtype=np.float32)
    edges = np.array([[[[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]]]], dtype=np.float32)

    w_corners = numpy_helper.from_array(corners, "w_corners")
    w_edges = numpy_helper.from_array(edges, "w_edges")

    sl_m5_starts = numpy_helper.from_array(np.array([5], dtype=np.int64), "sl_m5_starts")
    sl_m5_ends = numpy_helper.from_array(np.array([6], dtype=np.int64), "sl_m5_ends")
    sl_ax1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "sl_ax1")
    sum_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "sum_axes")

    sl9_starts = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), "sl9_starts")
    sl9_ends = numpy_helper.from_array(np.array([9, 9], dtype=np.int64), "sl9_ends")
    sl9_axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "sl9_axes")

    pad_amounts = numpy_helper.from_array(
        np.array([0, 0, 0, 0, 0, 0, 21, 21], dtype=np.int64), "pad_amounts")
    pad_value = numpy_helper.from_array(np.array([0.0], dtype=np.float32), "pad_value")

    nodes = []
    nodes.append(helper.make_node(
        "Slice", ["input", "sl_m5_starts", "sl_m5_ends", "sl_ax1"], ["m5"]))
    nodes.append(helper.make_node(
        "ReduceSum", ["input", "sum_axes"], ["in_grid"], keepdims=1))
    nodes.append(helper.make_node(
        "Conv", ["m5", "w_edges"], ["ch1_raw"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    nodes.append(helper.make_node(
        "Conv", ["m5", "w_corners"], ["ch5_raw"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    nodes.append(helper.make_node(
        "Slice", ["ch1_raw", "sl9_starts", "sl9_ends", "sl9_axes"], ["ch1_9"]))
    nodes.append(helper.make_node(
        "Slice", ["ch5_raw", "sl9_starts", "sl9_ends", "sl9_axes"], ["ch5_9"]))
    nodes.append(helper.make_node(
        "Pad", ["ch1_9", "pad_amounts", "pad_value"], ["ch1"], mode="constant"))
    nodes.append(helper.make_node(
        "Pad", ["ch5_9", "pad_amounts", "pad_value"], ["ch5"], mode="constant"))
    nodes.append(helper.make_node("Sub", ["in_grid", "ch1"], ["t0"]))
    nodes.append(helper.make_node("Sub", ["t0", "ch5"], ["ch0"]))
    nodes.append(helper.make_node("Sub", ["m5", "m5"], ["zero"]))
    nodes.append(helper.make_node(
        "Concat",
        ["ch0", "ch1", "zero", "zero", "zero", "ch5", "zero", "zero", "zero", "zero"],
        ["output"], axis=1))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task282",
        inputs=[inp], outputs=[out],
        initializer=[
            w_corners, w_edges,
            sl_m5_starts, sl_m5_ends, sl_ax1, sum_axes,
            sl9_starts, sl9_ends, sl9_axes,
            pad_amounts, pad_value,
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
    path = OUTPUT_DIR / "task282.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
