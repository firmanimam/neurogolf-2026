"""Build a hand-crafted static ONNX for task016 (fixed involution swap).

Rule (verified 267/267): channel permutation with involution
{1<->5, 2<->6, 3<->4, 8<->9}, 0 and 7 fixed.
Permutation vector p = [0, 5, 6, 4, 3, 1, 2, 7, 9, 8].

Static-ONNX recipe: single Gather(input, p, axis=1) with constant int64 p.
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
        np.array([0, 5, 6, 4, 3, 1, 2, 7, 9, 8], dtype=np.int64), "perm")

    nodes = [
        helper.make_node("Gather", ["input", "perm"], ["output"], axis=1),
    ]

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task016",
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
    path = OUTPUT_DIR / "task016.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
