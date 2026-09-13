#!/usr/bin/env python3
"""PTQ convert hey_siri streaming MFCC model to INT8 TFLite (kws_streaming utils)."""

import argparse
import json
import os
import sys
import wave
from argparse import Namespace

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.ops import gen_audio_ops as audio_ops

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


def _read_wav_pcm16(path, sample_rate, desired_samples):
  with wave.open(path, 'rb') as wav:
    if wav.getnchannels() != 1 or wav.getframerate() != sample_rate or wav.getsampwidth() != 2:
      return None
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
  if pcm.size < desired_samples:
    pcm = np.pad(pcm, (0, desired_samples - pcm.size))
  else:
    pcm = pcm[:desired_samples]
  return pcm


def _pcm_to_mfcc(pcm, flags):
  spectrogram = audio_ops.audio_spectrogram(
      tf.reshape(tf.convert_to_tensor(pcm, dtype=tf.float32), [-1, 1]),
      window_size=flags.window_size_samples,
      stride=flags.window_stride_samples,
      magnitude_squared=False)
  mfcc = audio_ops.mfcc(
      spectrogram=spectrogram,
      sample_rate=flags.sample_rate,
      upper_frequency_limit=flags.mel_upper_edge_hertz,
      lower_frequency_limit=flags.mel_lower_edge_hertz,
      filterbank_channel_count=flags.mel_num_bins,
      dct_coefficient_count=flags.dct_num_features)
  return mfcc.numpy()[0]


def _iter_calib_wavs(data_dir, keyword, max_files):
  rng = np.random.default_rng(42)
  keyword_dir = os.path.join(data_dir, keyword)
  siri = []
  if os.path.isdir(keyword_dir):
    siri.extend(
        os.path.join(keyword_dir, name)
        for name in os.listdir(keyword_dir)
        if name.lower().endswith('.wav'))
  others = []
  for name in os.listdir(data_dir):
    path = os.path.join(data_dir, name)
    if not os.path.isdir(path) or name.startswith('_') or name == keyword:
      continue
    for fn in os.listdir(path):
      if fn.lower().endswith('.wav'):
        others.append(os.path.join(path, fn))
  n_pos = max(1, min(len(siri), max_files // 2))
  n_neg = max(1, min(len(others), max_files - n_pos))
  selected = []
  if siri:
    idx = rng.choice(len(siri), size=n_pos, replace=False)
    selected.extend(siri[i] for i in idx)
  if others:
    idx = rng.choice(len(others), size=n_neg, replace=False)
    selected.extend(others[i] for i in idx)
  return selected[:max_files]


def collect_mfcc_frames(flags, data_dir, num_samples):
  """Real MFCC frames for PTQ. Random Gaussian scales destroy INT8 accuracy."""
  input_shape = modes.get_input_data_shape(
      flags, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
  keyword = flags.wanted_words.split(',')[0]
  wavs = _iter_calib_wavs(data_dir, keyword, max_files=max(24, num_samples // 10))
  frames = []
  for path in wavs:
    pcm = _read_wav_pcm16(path, flags.sample_rate, flags.desired_samples)
    if pcm is None:
      continue
    mfcc = _pcm_to_mfcc(pcm, flags)
    for frame in mfcc:
      frames.append(frame.reshape((1,) + tuple(input_shape)).astype(np.float32))
      if len(frames) >= num_samples:
        return frames
  if not frames:
    log(f'WARNING: no WAV calibration frames in {data_dir}, falling back to noise')
    rng = np.random.default_rng(42)
    frames = [
        rng.normal(loc=-5.0, scale=3.0, size=(1,) + tuple(input_shape)).astype(np.float32)
        for _ in range(num_samples)
    ]
  return frames


def collect_mfcc_clips(flags, data_dir, num_samples):
  """Full 1s MFCC clips for non-stream PTQ."""
  input_shape = modes.get_input_data_shape(flags, modes.Modes.NON_STREAM_INFERENCE)
  time_steps, feat_dim = input_shape
  keyword = flags.wanted_words.split(',')[0]
  wavs = _iter_calib_wavs(data_dir, keyword, max_files=max(num_samples, 24))
  clips = []
  for path in wavs:
    pcm = _read_wav_pcm16(path, flags.sample_rate, flags.desired_samples)
    if pcm is None:
      continue
    mfcc = _pcm_to_mfcc(pcm, flags)
    feat = np.zeros((time_steps, feat_dim), dtype=np.float32)
    n = min(mfcc.shape[0], time_steps)
    feat[-n:] = mfcc[:n]
    clips.append(feat.reshape((1,) + tuple(input_shape)))
    if len(clips) >= num_samples:
      break
  if not clips:
    log(f'WARNING: no WAV calibration clips in {data_dir}, falling back to noise')
    rng = np.random.default_rng(42)
    clips = [
        rng.normal(loc=-5.0, scale=3.0, size=(1,) + tuple(input_shape)).astype(np.float32)
        for _ in range(num_samples)
    ]
  return clips


def make_representative_dataset(flags, mode, num_samples=200, data_dir=''):
  input_shape = modes.get_input_data_shape(flags, mode)
  log(f'Representative dataset shape: {(1,) + tuple(input_shape)}')
  calib_dir = data_dir or getattr(flags, 'data_dir', '')
  if mode == modes.Modes.NON_STREAM_INFERENCE:
    samples = collect_mfcc_clips(flags, calib_dir, num_samples)
    log(f'Calibration clips: {len(samples)} from {calib_dir}')
  else:
    samples = collect_mfcc_frames(flags, calib_dir, num_samples)
    log(f'Calibration frames: {len(samples)} from {calib_dir}')

  def generator():
    for features in samples:
      yield [features]

  return generator


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      default='/home/wdf/kws_work/models/siri_bc_resnet2_run')
  parser.add_argument('--data_dir', default='',
                      help='WAV root for PTQ calibration. Defaults to flags.data_dir.')
  parser.add_argument('--rep_samples', type=int, default=400)
  parser.add_argument('--float', action='store_true',
                      help='Export float streaming TFLite without INT8 quantization.')
  parser.add_argument('--clip', action='store_true',
                      help='Export non-stream 1s clip model input [1, 50, 20].')
  args = parser.parse_args()

  flags = load_flags(args.model_dir)
  if flags.preprocess != 'mfcc':
    log(f'Expected preprocess=mfcc, got {flags.preprocess}')
    sys.exit(1)

  model = load_model(flags)
  if args.clip:
    convert_mode = modes.Modes.NON_STREAM_INFERENCE
    suffix = 'clip_float' if args.float else 'clip_int8'
  else:
    convert_mode = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
    suffix = 'float' if args.float else 'int8'
  saved_dir = os.path.join(flags.train_dir, f'stream_state_internal_{suffix}_saved')
  os.makedirs(saved_dir, exist_ok=True)

  if args.float:
    kind = 'float non-stream clip' if args.clip else 'float internal state'
    log(f'Converting with kws_streaming model_to_tflite ({kind})')
    tflite_bytes = utils.model_to_tflite(
        sess=None,
        model_non_stream=model,
        flags=flags,
        mode=convert_mode,
        save_model_path=saved_dir,
        optimizations=None,
        inference_type=tf1.lite.constants.FLOAT,
        experimental_new_quantizer=True,
        representative_dataset=None,
        supported_ops_override=[tf.lite.OpsSet.TFLITE_BUILTINS],
        allow_custom_ops=False,
        inference_input_type=tf.float32,
        inference_output_type=tf.float32,
    )
    out_name = 'stream_state_internal.tflite'
  else:
    # Clip BC-ResNet: full INT8 PTQ (activations) drops siri recall to ~30%.
    # Weight-only INT8 keeps float accuracy. Streaming still uses PTQ calib.
    use_ptq = not args.clip
    kind = 'INT8 weight-only clip' if args.clip else 'INT8 internal state PTQ'
    log(f'Converting with kws_streaming model_to_tflite ({kind})')
    tflite_bytes = utils.model_to_tflite(
        sess=None,
        model_non_stream=model,
        flags=flags,
        mode=convert_mode,
        save_model_path=saved_dir,
        optimizations=[tf.lite.Optimize.DEFAULT],
        inference_type=tf1.lite.constants.FLOAT,
        experimental_new_quantizer=True,
        representative_dataset=(
            make_representative_dataset(
                flags, convert_mode, args.rep_samples, args.data_dir)
            if use_ptq else None),
        supported_ops_override=[tf.lite.OpsSet.TFLITE_BUILTINS],
        allow_custom_ops=False,
        inference_input_type=tf.float32,
        inference_output_type=tf.float32,
    )
    out_name = 'stream_state_internal_int8.tflite'

  out_dir = os.path.join(args.model_dir, 'tflite_stream_state_internal')
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, out_name)
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
