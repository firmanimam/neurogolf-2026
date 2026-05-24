"""Build a hand-crafted static ONNX for task004 (parallelogram shift-right-with-clamp).

Rule (verified 265/265): for each non-zero color in the input grid, the cells
forming a parallelogram outline. The bottom row stays put. All other cells get
col -> col+1, clamped to the bottom row's max column.

Static-ONNX recipe (no dynamic indexing — purely tensor ops on a sliced
(1,9,30,30) of nonzero-color channels):

  inz       = Slice(input, ch=1:10)                    # (1,9,30,30)
  # 1) Bottom-row mask: per color, 1 only at r == max row that has color.
  row_pres  = ReduceSum(inz, axes=[3])                 # (1,9,30)
  row_cs    = CumSum(row_pres, axis=2, reverse=1)      # sum over r' >= r
  any_below = Cast(Greater(row_cs, 0))                 # 1 if any cell at r' >= r
  next_any  = Pad-shift any_below "up" by 1 in r       # any_below[r+1]
  bot_mask  = any_below - next_any                     # 1 at bottom row, else 0
  bot_4d    = Reshape bot_mask -> (1,9,30,1)
  bot_layer = inz * bot_4d                             # (1,9,30,30) bottom cells
  nonbot    = inz - bot_layer
  # 2) Right-shift nonbottom by 1 col.
  shifted   = Pad(nonbot, [0,0,0,1, 0,0,0,0]) sliced [..., :30]  # (1,9,30,30)
  # 3) Inside-mask: per color, 1 if col <= max_col_c.
  col_pres  = ReduceSum(inz, axes=[2])                 # (1,9,30)
  col_cs    = CumSum(col_pres, axis=2, reverse=1)
  in_col    = Cast(Greater(col_cs, 0))                 # 1 if col <= max_col_c
  next_in   = Pad-shift in_col "left" by 1 in col      # in_col[col+1]
  at_max    = in_col - next_in                         # 1 at col == max_col_c
  in_col_4d = Reshape in_col -> (1,9,1,30)
  at_max_4d = Reshape at_max -> (1,9,1,30)
  out_mask_4d = 1 - in_col_4d
  trimmed   = shifted * in_col_4d                      # zero out beyond max_col
  overflow_cells = shifted * out_mask_4d               # only at col = max_col_c+1
  overflow_row   = ReduceSum(overflow_cells, axes=[3], keepdims=1)  # (1,9,30,1)
  overflow_add   = overflow_row * at_max_4d            # broadcast to (1,9,30,30)
  output_nz      = trimmed + overflow_add + bot_layer  # (1,9,30,30)
  # 4) Channel 0 = in_grid AND not in any nonzero output.
  in_grid   = ReduceSum(input, axes=[1], keepdims=1)   # (1,1,30,30) always 0/1
  any_nz    = ReduceSum(output_nz, axes=[1], keepdims=1)
  ch0       = in_grid - any_nz                         # (1,1,30,30)
  output    = Concat([ch0, output_nz], axis=1)         # (1,10,30,30)
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
    # ── Initializers ─────────────────────────────────────────────────────
    # Slice channels 1..9
    s_ch_start = numpy_helper.from_array(np.array([1], dtype=np.int64), "s_ch_start")
    s_ch_end = numpy_helper.from_array(np.array([10], dtype=np.int64), "s_ch_end")
    s_ch_axis = numpy_helper.from_array(np.array([1], dtype=np.int64), "s_ch_axis")

    # Reductions
    axes_3 = numpy_helper.from_array(np.array([3], dtype=np.int64), "axes_3")
    axes_2 = numpy_helper.from_array(np.array([2], dtype=np.int64), "axes_2")
    axes_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_1")

    # Constants
    zero_int = numpy_helper.from_array(np.array(0, dtype=np.int64), "zero_int")
    one_f = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_f")

    # Reshape shapes
    shape_1_9_30_1 = numpy_helper.from_array(
        np.array([1, 9, 30, 1], dtype=np.int64), "shape_1_9_30_1")
    shape_1_9_1_30 = numpy_helper.from_array(
        np.array([1, 9, 1, 30], dtype=np.int64), "shape_1_9_1_30")

    # Pad/Slice for shifting along col axis (right by 1):
    # pad left=1 on W (axis 3), then slice [0:30] on axis 3.
    pad_value_f = numpy_helper.from_array(np.array([0.0], dtype=np.float32), "pad_value_f")
    s_col_axis = numpy_helper.from_array(np.array([3], dtype=np.int64), "s_col_axis")
    s_col_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "s_col_0")
    s_col_29 = numpy_helper.from_array(np.array([29], dtype=np.int64), "s_col_29")
    zero_col_4d = numpy_helper.from_array(
        np.zeros((1, 9, 30, 1), dtype=np.float32), "zero_col_4d")

    # Shift "left by 1" for a 3D (1,9,30) tensor along axis=2:
    # take [1:30] (axis 2), then pad right=1 zero.
    s_axis2 = numpy_helper.from_array(np.array([2], dtype=np.int64), "s_axis2")
    s_1_30_start = numpy_helper.from_array(np.array([1], dtype=np.int64), "s_1_30_start")
    s_1_30_end = numpy_helper.from_array(np.array([30], dtype=np.int64), "s_1_30_end")
    pads_3d_right = numpy_helper.from_array(
        np.array([0, 0, 0, 0, 0, 1], dtype=np.int64), "pads_3d_right")

    nodes = []

    # ── Slice non-zero color channels ─────────────────────────────────────
    nodes.append(helper.make_node(
        "Slice", ["input", "s_ch_start", "s_ch_end", "s_ch_axis"], ["inz"]))

    # ── Bottom row mask ───────────────────────────────────────────────────
    # row_pres = ReduceSum(inz, axes=[3], keepdims=0) → (1,9,30)
    nodes.append(helper.make_node(
        "ReduceSum", ["inz", "axes_3"], ["row_pres"], keepdims=0))
    # row_cs = CumSum(row_pres, axis=2, reverse=1)
    nodes.append(helper.make_node(
        "CumSum", ["row_pres", "axes_2_scalar_for_cumsum_r"], ["row_cs"],
        reverse=1))
    # We need an int64 SCALAR axis for CumSum. Use zero-d tensor "axis_r=2".
    # Re-emit with proper initializer:

    # any_below = Cast(Greater(row_cs, 0)) → float
    nodes.append(helper.make_node(
        "Greater", ["row_cs", "zero_f_scalar"], ["any_below_bool"]))
    nodes.append(helper.make_node(
        "Cast", ["any_below_bool"], ["any_below"], to=TensorProto.FLOAT))

    # next_any = shift "up" by 1 in r axis (axis=2 of 3D tensor):
    #   Slice rows [1:30] then Pad right=1
    nodes.append(helper.make_node(
        "Slice", ["any_below", "s_1_30_start", "s_1_30_end", "s_axis2"],
        ["any_below_sliced"]))
    nodes.append(helper.make_node(
        "Pad", ["any_below_sliced", "pads_3d_right", "pad_value_f"],
        ["next_any"], mode="constant"))

    # bot_mask = any_below - next_any → (1,9,30)
    nodes.append(helper.make_node("Sub", ["any_below", "next_any"], ["bot_mask"]))

    # Reshape bot_mask -> (1,9,30,1)
    nodes.append(helper.make_node(
        "Reshape", ["bot_mask", "shape_1_9_30_1"], ["bot_4d"]))

    # bot_layer = inz * bot_4d (broadcast on last axis)
    nodes.append(helper.make_node("Mul", ["inz", "bot_4d"], ["bot_layer"]))

    # nonbot = inz - bot_layer
    nodes.append(helper.make_node("Sub", ["inz", "bot_layer"], ["nonbot"]))

    # ── Shift nonbot right by 1 col (Slice [:,:,:,:29] then Concat zero-col) ─
    nodes.append(helper.make_node(
        "Slice", ["nonbot", "s_col_0", "s_col_29", "s_col_axis"], ["nonbot_29"]))
    nodes.append(helper.make_node(
        "Concat", ["zero_col_4d", "nonbot_29"], ["shifted"], axis=3))

    # ── Inside mask (col <= max_col_c) ────────────────────────────────────
    nodes.append(helper.make_node(
        "ReduceSum", ["inz", "axes_2"], ["col_pres"], keepdims=0))  # (1,9,30)
    nodes.append(helper.make_node(
        "CumSum", ["col_pres", "axes_2_scalar_for_cumsum_c"], ["col_cs"],
        reverse=1))
    nodes.append(helper.make_node(
        "Greater", ["col_cs", "zero_f_scalar"], ["in_col_bool"]))
    nodes.append(helper.make_node(
        "Cast", ["in_col_bool"], ["in_col"], to=TensorProto.FLOAT))

    # next_in = shift "left" by 1 in col axis (axis=2 of 3D): Slice [1:30], Pad right=1
    nodes.append(helper.make_node(
        "Slice", ["in_col", "s_1_30_start", "s_1_30_end", "s_axis2"],
        ["in_col_sliced"]))
    nodes.append(helper.make_node(
        "Pad", ["in_col_sliced", "pads_3d_right", "pad_value_f"],
        ["next_in"], mode="constant"))

    # at_max = in_col - next_in
    nodes.append(helper.make_node("Sub", ["in_col", "next_in"], ["at_max"]))

    # Reshape to 4D
    nodes.append(helper.make_node(
        "Reshape", ["in_col", "shape_1_9_1_30"], ["in_col_4d"]))
    nodes.append(helper.make_node(
        "Reshape", ["at_max", "shape_1_9_1_30"], ["at_max_4d"]))

    # trimmed = shifted * in_col_4d
    nodes.append(helper.make_node("Mul", ["shifted", "in_col_4d"], ["trimmed"]))

    # overflow_row = ReduceSum(nonbot * at_max_4d, axes=[3], keepdims=1)
    # (cells in nonbot at col=max_col_c are exactly the ones that overflow)
    nodes.append(helper.make_node(
        "Mul", ["nonbot", "at_max_4d"], ["overflow_cells"]))
    nodes.append(helper.make_node(
        "ReduceSum", ["overflow_cells", "axes_3"], ["overflow_row"], keepdims=1))

    # overflow_add = overflow_row * at_max_4d (broadcast 30x1 × 1x30 → 30x30)
    nodes.append(helper.make_node(
        "Mul", ["overflow_row", "at_max_4d"], ["overflow_add"]))

    # output_nz = trimmed + overflow_add + bot_layer  (single Sum op)
    nodes.append(helper.make_node(
        "Sum", ["trimmed", "overflow_add", "bot_layer"], ["output_nz"]))

    # ── Channel 0: in_grid - any_nz ───────────────────────────────────────
    nodes.append(helper.make_node(
        "ReduceSum", ["input", "axes_1"], ["in_grid"], keepdims=1))   # (1,1,30,30)
    nodes.append(helper.make_node(
        "ReduceSum", ["output_nz", "axes_1"], ["any_nz"], keepdims=1))
    nodes.append(helper.make_node("Sub", ["in_grid", "any_nz"], ["ch0"]))

    # ── Concat output ─────────────────────────────────────────────────────
    nodes.append(helper.make_node(
        "Concat", ["ch0", "output_nz"], ["output"], axis=1))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    # Scalar zero float for Greater
    zero_f_scalar = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_f_scalar")
    # Scalar int64 axis=2 for CumSum (CumSum needs a 0-D int tensor)
    axis2_scalar_r = numpy_helper.from_array(
        np.array(2, dtype=np.int64), "axes_2_scalar_for_cumsum_r")
    axis2_scalar_c = numpy_helper.from_array(
        np.array(2, dtype=np.int64), "axes_2_scalar_for_cumsum_c")

    graph = helper.make_graph(
        nodes, "task004",
        inputs=[inp], outputs=[out],
        initializer=[
            s_ch_start, s_ch_end, s_ch_axis,
            axes_3, axes_2, axes_1,
            zero_int, one_f, zero_f_scalar,
            shape_1_9_30_1, shape_1_9_1_30,
            pad_value_f, s_col_axis, s_col_0, s_col_29, zero_col_4d,
            s_axis2, s_1_30_start, s_1_30_end,
            pads_3d_right,
            axis2_scalar_r, axis2_scalar_c,
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
    path = OUTPUT_DIR / "task004.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
