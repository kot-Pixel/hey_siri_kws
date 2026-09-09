#!/usr/bin/env python3
"""PTQ convert hey_siri streaming MFCC model to INT8 TFLite (kws_streaming utils)."""

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from kws_streaming.layers import modes
from kws_streaming.models import model_flags, utils

tf.compat.v1.enable_eager_execution()


def log(msg):
  print(msg, flush=True)


def load_flags(model_dir):
  with open(os.path.join(model_dir, 'flags.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
  flags = Namespace(**data)
  flags.training = False
  flags.batch_size = 1
  flags.train_dir = model_dir
  return model_flags.update_flags(flags)


def _import_model_module(model_name):
  """Dynamically import the model builder from kws_streaming.models."""
  import importlib
  module = importlib.import_module(f'kws_streaming.models.{model_name}')
  return module


def load_model(flags):
  model_name = flags.model_name if hasattr(flags, 'model_name') else 'svdf'
  log(f'Loading model type: {model_name}')
  mod = _import_model_module(model_name)
  model = mod.model(flags)
  model.load_weights(os.path.join(flags.train_dir, 'best_weights')).expect_partial()
  return model


def make_representative_dataset(flags, num_samples=200):
  rng = np.random.default_rng(42)
  input_shape = modes.get_input_data_shape(flags, modes.Modes.TRAINING)

  def generator():
    for _ in range(num_samples):
      features = rng.normal(loc=-5.0, scale=3.0,
                            size=(1,) + tuple(input_shape)).astype(np.float32)
      yield [features]

  return generator


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      default='/home/wdf/kws_work/models/hey_siri_mfcc_run')
  parser.add_argument('--rep_samples', type=int, default=200)
  args = parser.parse_args()

  flags = load_flags(args.model_dir)
  if flags.preprocess != 'mfcc':
    log(f'Expected preprocess=mfcc, got {flags.preprocess}')
    sys.exit(1)

  model = load_model(flags)
  saved_dir = os.path.join(flags.train_dir, 'stream_state_internal_int8_saved')
  os.makedirs(saved_dir, exist_ok=True)

  log('Converting with kws_streaming model_to_tflite (INT8, internal state)')
  tflite_bytes = utils.model_to_tflite(
      sess=None,
      model_non_stream=model,
      flags=flags,
      mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
      save_model_path=saved_dir,
      optimizations=[tf.lite.Optimize.DEFAULT],
      inference_type=tf1.lite.constants.QUANTIZED_UINT8,
      experimental_new_quantizer=False,
      representative_dataset=make_representative_dataset(flags, args.rep_samples),
      supported_ops_override=[tf.lite.OpsSet.TFLITE_BUILTINS],
      allow_custom_ops=False,
      inference_input_type=tf.float32,
      inference_output_type=tf.float32,
  )

  out_dir = os.path.join(args.model_dir, 'tflite_stream_state_internal')
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, 'stream_state_internal_int8.tflite')
  with open(out_path, 'wb') as f:
    f.write(tflite_bytes)
  log(f'Wrote {out_path} ({len(tflite_bytes)} bytes)')

  interp = tf.lite.Interpreter(model_path=out_path)
  interp.allocate_tensors()
  for detail in interp.get_input_details():
    log(f'input: name={detail["name"]} shape={detail["shape"]} dtype={detail["dtype"]}')
  for detail in interp.get_output_details():
    log(f'output: name={detail["name"]} shape={detail["shape"]} dtype={detail["dtype"]}')


if __name__ == '__main__':
  main()
