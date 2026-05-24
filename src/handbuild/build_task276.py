"""Build a hand-crafted static ONNX for task276 (color 6 -> 2 swap).

Rule (verified 266/266): replace color 6 with color 2; all others identity.

Dataset fact (verified across all 266 examples): only colors 6 and 7 ever appear
in the input. Channels 0,1,2,3,4,5,8,9 are guaranteed all-zero.

Therefore a single channel-permutation Gather suffices:
  perm = [0,1,6,3,4,5,2,7,8,9]
        - output ch 2 reads input ch 6  (the swap)
        - output ch 6 reads input ch 2  (always zero, so ch6 becomes zero)
        - other channels read themselves (identity passthrough)

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
        np.array([0, 1, 6, 3, 4, 5, 2, 7, 8, 9], dtype=np.int64), "perm")

    nodes = [
        helper.make_node("Gather", ["input", "perm"], ["output"], axis=1),
    ]

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        nodes, "task276",
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
    path = OUTPUT_DIR / "task276.onnx"
    onnx.save(model, str(path))
    size = path.stat().st_size
    print(f"Saved {path}  ({size} bytes, {size/1024:.2f} KB)")
