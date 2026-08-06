#!/usr/bin/env python3
"""Inspect TensorRT parser layer names/types and tensor dtypes."""

import argparse
import json
from pathlib import Path

import tensorrt as trt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--contains", default="rotary")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = [str(onnx_parser.get_error(index)) for index in range(onnx_parser.num_errors)]
        raise SystemExit("\n".join(errors))
    result = []
    needle = args.contains.lower()
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        selected = needle in layer.name.lower()
        if args.end > args.start:
            selected = selected or args.start <= index <= args.end
        if not selected:
            continue
        inputs = []
        for input_index in range(layer.num_inputs):
            tensor = layer.get_input(input_index)
            inputs.append(None if tensor is None else {"name": tensor.name, "dtype": str(tensor.dtype)})
        outputs = []
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            outputs.append(None if tensor is None else {"name": tensor.name, "dtype": str(tensor.dtype)})
        result.append(
            {
                "index": index,
                "name": layer.name,
                "type": str(layer.type),
                "inputs": inputs,
                "outputs": outputs,
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
