"""Build a hand-crafted static ONNX for task337 (swap channels 5 and 8).

Rule (verified 266/266): identity on all channels except swap channels 5 and 8.

Static-ONNX recipe — pure channel permutation via single Gather:
  perm   = [0,1,2,3,4,8,6,7,5,9]  (int64 constant of shape (10,))
  output = Gather(input, perm, axis=1)
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
    perm = numpy_helper.from_array(
        np.array([0, 1, 2, 3, 4, 8, 6, 7, 5, 9], dtype=np.int64), "perm")

    nodes = [
        helper.make_node("Gather", ["input", "perm"], ["output"], axis=1),
    ]

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task337",
        inputs=[inp], outputs=[out],
        initializer=[perm],
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
    path = OUTPUT_DIR / "task337.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
