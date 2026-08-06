#!/usr/bin/env python3
"""Promote FP16 ONNX MatMul inputs to FP32 and cast each result back to FP16."""

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, data_prop=False)
    types = {initializer.name: initializer.data_type for initializer in inferred.graph.initializer}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    return types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load_model(args.input, load_external_data=True)
    types = tensor_types(model)
    rewritten = []
    promoted = 0
    for index, node in enumerate(model.graph.node):
        if (
            node.op_type != "MatMul"
            or len(node.input) != 2
            or len(node.output) != 1
            or types.get(node.input[0]) != TensorProto.FLOAT16
            or types.get(node.input[1]) != TensorProto.FLOAT16
        ):
            rewritten.append(node)
            continue
        left = f"{node.input[0]}__matmul_fp32_left_{index}"
        right = f"{node.input[1]}__matmul_fp32_right_{index}"
        fp32_output = f"{node.output[0]}__matmul_fp32_{index}"
        rewritten.extend(
            [
                helper.make_node("Cast", [node.input[0]], [left], to=TensorProto.FLOAT),
                helper.make_node("Cast", [node.input[1]], [right], to=TensorProto.FLOAT),
                helper.make_node("MatMul", [left, right], [fp32_output], name=node.name),
                helper.make_node("Cast", [fp32_output], [node.output[0]], to=TensorProto.FLOAT16),
            ]
        )
        promoted += 1
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_name = args.output.name + ".data"
    data_path = args.output.parent / data_name
    data_path.unlink(missing_ok=True)
    onnx.save_model(
        model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(args.output))
    print(json.dumps({"input": str(args.input), "output": str(args.output), "promoted_matmuls": promoted}))


if __name__ == "__main__":
    main()
