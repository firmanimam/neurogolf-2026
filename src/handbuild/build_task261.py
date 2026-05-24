"""task261 — static shift down by 1 + recolor 8→2.

Static ONNX recipe (no dynamic channel detection needed):
  input_8       = Slice(input, [8:9], axis=1)             # (1,1,30,30)
  cropped       = Slice(input_8, [0:29], axis=2)          # (1,1,29,30)
  shifted_8     = Pad(cropped, pads=[0,0,1,0,0,0,0,0])    # (1,1,30,30)  shift down 1
  in_grid       = ReduceMax(input, axes=[1], keepdims=1)  # (1,1,30,30)  1 where any color
  zeros_2d      = ConstantOfShape([1,30,30], 0)            # (1,30,30)
  out_ch0       = in_grid - shifted_8 (mul, since both 0/1)   wait, must be AND
  out_ch0       = in_grid * (1 - shifted_8)               # (1,1,30,30)
  output_zeros  = ConstantOfShape([1,10,30,30], 0)
  output        = ScatterND(output_zeros, [[0,0],[0,2]], [out_ch0_3d, shifted_8_3d])
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
    axes_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_1")
    shape_full = numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64), "shape_full")
    shape_1_30_30 = numpy_helper.from_array(np.array([1, 30, 30], dtype=np.int64), "shape_1_30_30")
    # Slice channel 8
    slice_8_starts = numpy_helper.from_array(np.array([8], dtype=np.int64), "slice_8_starts")
    slice_8_ends = numpy_helper.from_array(np.array([9], dtype=np.int64), "slice_8_ends")
    slice_axis_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "slice_axis_1")
    # Slice rows 0:29 (axis 2)
    slice_rows_starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_rows_starts")
    slice_rows_ends = numpy_helper.from_array(np.array([29], dtype=np.int64), "slice_rows_ends")
    slice_axis_2 = numpy_helper.from_array(np.array([2], dtype=np.int64), "slice_axis_2")
    # Pad amounts: 8 values for (start_N, start_C, start_H, start_W, end_N, end_C, end_H, end_W)
    pad_amounts = numpy_helper.from_array(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64), "pad_amounts")
    pad_value = numpy_helper.from_array(np.array([0.0], dtype=np.float32), "pad_value")
    # Constants
    one_scalar = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_scalar")
    # ConstantOfShape value
    cos_value = helper.make_tensor("cos_value", TensorProto.FLOAT, [1], [0.0])
    # Scatter indices for output: channel 0 and channel 2
    scatter_indices = numpy_helper.from_array(np.array([[0, 0], [0, 2]], dtype=np.int64), "scatter_indices")

    nodes = []
    # 1) input_8 = Slice(input, [8:9], axis=1)
    nodes.append(helper.make_node(
        "Slice", ["input", "slice_8_starts", "slice_8_ends", "slice_axis_1"], ["input_8"]))
    # 2) cropped = Slice(input_8, [0:29], axis=2)
    nodes.append(helper.make_node(
        "Slice", ["input_8", "slice_rows_starts", "slice_rows_ends", "slice_axis_2"], ["cropped"]))
    # 3) shifted_8 = Pad(cropped, pad_amounts, 0.0)
    nodes.append(helper.make_node(
        "Pad", ["cropped", "pad_amounts", "pad_value"], ["shifted_8"], mode="constant"))

    # 4) in_grid = ReduceMax(input, axes=[1], keepdims=1)  → (1,1,30,30)
    # NOTE: in opset 17, ReduceMax takes axes as attribute (not input as in opset 18+)
    nodes.append(helper.make_node(
        "ReduceMax", ["input"], ["in_grid"], axes=[1], keepdims=1))

    # 5) inv_shifted = 1 - shifted_8 (using Sub)
    nodes.append(helper.make_node("Sub", ["one_scalar", "shifted_8"], ["inv_shifted"]))
    # 6) out_ch0 = in_grid * inv_shifted
    nodes.append(helper.make_node("Mul", ["in_grid", "inv_shifted"], ["out_ch0"]))

    # 7) Reshape out_ch0 and shifted_8 to (1,30,30) for ScatterND updates
    nodes.append(helper.make_node("Reshape", ["out_ch0", "shape_1_30_30"], ["upd_ch0"]))
    nodes.append(helper.make_node("Reshape", ["shifted_8", "shape_1_30_30"], ["upd_ch2"]))
    # 8) Concat updates → (2, 30, 30)
    nodes.append(helper.make_node("Concat", ["upd_ch0", "upd_ch2"], ["updates"], axis=0))

    # 9) Build output via ScatterND on a zeros tensor
    nodes.append(helper.make_node(
        "ConstantOfShape", ["shape_full"], ["zeros_full"], value=cos_value))
    nodes.append(helper.make_node(
        "ScatterND", ["zeros_full", "scatter_indices", "updates"], ["output"]))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task261",
        inputs=[inp], outputs=[out],
        initializer=[
            axes_1, shape_full, shape_1_30_30,
            slice_8_starts, slice_8_ends, slice_axis_1,
            slice_rows_starts, slice_rows_ends, slice_axis_2,
            pad_amounts, pad_value,
            one_scalar,
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
    path = OUTPUT_DIR / "task261.onnx"
    onnx.save(model, str(path))
    print(f"Saved {path}  ({path.stat().st_size} bytes)")
