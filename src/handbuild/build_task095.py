"""task095 — 3x3 dilation around color-5, fill neighbors with 1, keep 5 at center.

Static ONNX recipe:
  input_5   = Slice(input, [5:6], axis=1)              # (1,1,30,30)
  dilated   = Conv(input_5, ones_3x3, pad=1)           # (1,1,30,30) count of 5s in 3x3
  in_grid   = ReduceMax(input, axes=[1], keepdims=1)   # (1,1,30,30)
  has_neigh = Clip(dilated, 0, 1) - input_5            # 1 where neighbor 5 exists and self != 5
  has_neigh = Clip(has_neigh, 0, 1)                    # ensure 0/1 (subtraction may produce 0)
  ch1       = has_neigh * in_grid                      # mask to active grid
  ch5       = input_5
  not_near  = (1 - Clip(dilated, 0, 1)) * in_grid      # cells in-grid with no 5 nearby
  ch0       = not_near
  output    = ScatterND(zeros, [[0,0],[0,1],[0,5]], [ch0,ch1,ch5])
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
    # Initializers
    shape_full = numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64), "shape_full")
    shape_1_30_30 = numpy_helper.from_array(np.array([1, 30, 30], dtype=np.int64), "shape_1_30_30")
    slice_5_starts = numpy_helper.from_array(np.array([5], dtype=np.int64), "slice_5_starts")
    slice_5_ends = numpy_helper.from_array(np.array([6], dtype=np.int64), "slice_5_ends")
    slice_axis_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "slice_axis_1")
    one_scalar = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_scalar")
    # Conv weights: 3x3 ones, shape (1, 1, 3, 3)
    conv_w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), "conv_w")
    # Clip min/max scalars
    clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min")
    clip_max = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max")
    # Scatter indices: channels 0, 1, 5
    scatter_indices = numpy_helper.from_array(np.array([[0, 0], [0, 1], [0, 5]], dtype=np.int64), "scatter_indices")
    cos_value = helper.make_tensor("cos_value", TensorProto.FLOAT, [1], [0.0])

    nodes = []
    # 1) input_5 = Slice(input, [5:6], axis=1) → (1,1,30,30)
    nodes.append(helper.make_node(
        "Slice", ["input", "slice_5_starts", "slice_5_ends", "slice_axis_1"], ["input_5"]))
    # 2) dilated = Conv(input_5, conv_w, pad=1) → (1,1,30,30)
    nodes.append(helper.make_node(
        "Conv", ["input_5", "conv_w"], ["dilated"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1], group=1))
    # 3) dilated_bool = Clip(dilated, 0, 1) → (1,1,30,30)
    nodes.append(helper.make_node(
        "Clip", ["dilated", "clip_min", "clip_max"], ["dilated_bool"]))
    # 4) in_grid = ReduceMax(input, axes=[1], keepdims=1) → (1,1,30,30)
    nodes.append(helper.make_node(
        "ReduceMax", ["input"], ["in_grid"], axes=[1], keepdims=1))

    # 5) Compute channel 1: (dilated_bool - input_5) clipped to [0,1], then * in_grid
    nodes.append(helper.make_node("Sub", ["dilated_bool", "input_5"], ["neigh_raw"]))
    nodes.append(helper.make_node(
        "Clip", ["neigh_raw", "clip_min", "clip_max"], ["neigh_clipped"]))
    nodes.append(helper.make_node("Mul", ["neigh_clipped", "in_grid"], ["ch1"]))

    # 6) Compute channel 0: (1 - dilated_bool) * in_grid
    nodes.append(helper.make_node("Sub", ["one_scalar", "dilated_bool"], ["not_dilated"]))
    nodes.append(helper.make_node("Mul", ["not_dilated", "in_grid"], ["ch0"]))

    # 7) Channel 5 = input_5 (alias)
    # We'll just reshape input_5 for scatter

    # 8) Reshape each to (1,30,30) and concat → (3,30,30)
    nodes.append(helper.make_node("Reshape", ["ch0", "shape_1_30_30"], ["upd_ch0"]))
    nodes.append(helper.make_node("Reshape", ["ch1", "shape_1_30_30"], ["upd_ch1"]))
    nodes.append(helper.make_node("Reshape", ["input_5", "shape_1_30_30"], ["upd_ch5"]))
    nodes.append(helper.make_node("Concat", ["upd_ch0", "upd_ch1", "upd_ch5"], ["updates"], axis=0))

    # 9) Output via ScatterND
    nodes.append(helper.make_node(
        "ConstantOfShape", ["shape_full"], ["zeros_full"], value=cos_value))
    nodes.append(helper.make_node(
        "ScatterND", ["zeros_full", "scatter_indices", "updates"], ["output"]))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task095",
        inputs=[inp], outputs=[out],
        initializer=[
            shape_full, shape_1_30_30,
            slice_5_starts, slice_5_ends, slice_axis_1,
            one_scalar, conv_w,
            clip_min, clip_max,
            scatter_indices,
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
    path = OUTPUT_DIR / "task095.onnx"
    onnx.save(model, str(path))
    print(f"Saved {path}  ({path.stat().st_size} bytes)")
