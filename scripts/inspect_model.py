#!/usr/bin/env python3
"""Print TFLite model input/output tensor details."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=args.model)
    except ImportError:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=args.model)

    interp.allocate_tensors()
    print("INPUTS")
    for detail in interp.get_input_details():
        print(detail)
    print("OUTPUTS")
    for detail in interp.get_output_details():
        print(detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
