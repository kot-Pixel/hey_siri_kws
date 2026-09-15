#!/usr/bin/env python3
"""Evaluate KWS model on a folder-per-label WAV dataset.

wake mode matches C++ realtime logic in src/hey_siri_kws.cpp:
  50-frame MFCC sliding window, invoke every KWS_INFER_STRIDE frames,
  softmax(siri) >= KWS_WAKE_THRESHOLD for KWS_WAKE_HITS consecutive infers.
"""

import argparse
import hashlib
import os
import re
import sys
import wave

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops


SIRI_INDEX = 2
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1


def log(msg):
    print(msg, flush=True)


def which_set(filename, validation_percentage=10, testing_percentage=10):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    percentage_hash = (
        (int(hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
        * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    )
    if percentage_hash < validation_percentage:
        return "validation"
    if percentage_hash < testing_percentage + validation_percentage:
        return "testing"
    return "training"


def read_wav_pcm16(path, sample_rate=16000, min_samples=0, truncate=False):
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("not mono")
        if wav.getframerate() != sample_rate:
            raise ValueError("bad rate")
        if wav.getsampwidth() != 2:
            raise ValueError("not s16")
        n = wav.getnframes()
        raw = wav.readframes(n)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if min_samples > 0 and pcm.size < min_samples:
        pcm = np.pad(pcm, (0, min_samples - pcm.size))
    elif truncate and min_samples > 0 and pcm.size > min_samples:
        pcm = pcm[:min_samples]
    return pcm


def pcm_to_mfcc(pcm, sample_rate, window_size, window_stride, mel_lo, mel_hi, mel_bins, dct_bins):
    spectrogram = audio_ops.audio_spectrogram(
        tf.reshape(tf.convert_to_tensor(pcm, dtype=tf.float32), [-1, 1]),
        window_size=window_size,
        stride=window_stride,
        magnitude_squared=False,
    )
    mfcc = audio_ops.mfcc(
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        upper_frequency_limit=mel_hi,
        lower_frequency_limit=mel_lo,
        filterbank_channel_count=mel_bins,
        dct_coefficient_count=dct_bins,
    )
    return mfcc.numpy()[0]


def list_dataset(data_dir, keyword, max_per_label, max_keyword=0, split="all"):
    labels = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path) or name.startswith("_"):
            continue
        wavs = [
            os.path.join(path, fn)
            for fn in sorted(os.listdir(path))
            if fn.lower().endswith(".wav")
        ]
        if split != "all":
            wavs = [p for p in wavs if which_set(p) == split]
        if not wavs:
            continue
        if name == keyword and max_keyword > 0:
            wavs = wavs[:max_keyword]
        elif name != keyword and max_per_label > 0:
            wavs = wavs[:max_per_label]
        labels.append((name, wavs))
    return labels


def softmax(logits):
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / np.sum(e)


def run_stream(interp, input_index, output_index, mfcc):
    last = None
    max_siri = -1e9
    max_siri_prob = 0.0
    for frame in mfcc:
        interp.set_tensor(input_index, frame.reshape(1, 1, -1).astype(np.float32))
        interp.invoke()
        last = interp.get_tensor(output_index)[0]
        max_siri = max(max_siri, float(last[SIRI_INDEX]))
        max_siri_prob = max(max_siri_prob, float(softmax(last)[SIRI_INDEX]))
    return last, max_siri, max_siri_prob


def load_keras_model(model_dir):
    import importlib
    import json
    from argparse import Namespace

    from kws_streaming.models import model_flags

    with open(os.path.join(model_dir, "flags.json"), encoding="utf-8") as f:
        flags = Namespace(**json.load(f))
    flags.training = False
    flags.batch_size = 1
    flags.train_dir = model_dir
    flags = model_flags.update_flags(flags)
    mod = importlib.import_module(f"kws_streaming.models.{flags.model_name}")
    model = mod.model(flags)
    model.load_weights(os.path.join(model_dir, "best_weights")).expect_partial()
    return model


def run_clip_tflite(interp, input_index, output_index, mfcc, time_steps, feat_dim):
    feat = np.zeros((time_steps, feat_dim), dtype=np.float32)
    n = min(mfcc.shape[0], time_steps)
    feat[-n:] = mfcc[:n]
    interp.set_tensor(input_index, feat.reshape(1, time_steps, feat_dim))
    interp.invoke()
    last = interp.get_tensor(output_index)[0]
    prob = float(softmax(last)[SIRI_INDEX])
    return last, float(last[SIRI_INDEX]), prob


def run_clip_wake(interp, input_index, output_index, mfcc, time_steps, feat_dim,
                  infer_stride, wake_threshold, wake_hits):
    """Slide a 1s clip window like hey_siri_kws.cpp FeedFrame + kws_is_wake."""
    window = np.zeros((time_steps, feat_dim), dtype=np.float32)
    frames_until = 0
    hits = 0
    woke = False
    max_prob = 0.0
    last = np.zeros((3,), dtype=np.float32)
    n_infer = 0
    for frame in mfcc:
        window[:-1] = window[1:]
        window[-1] = frame
        if frames_until > 0:
            frames_until -= 1
            continue
        frames_until = infer_stride - 1
        interp.set_tensor(input_index, window.reshape(1, time_steps, feat_dim))
        interp.invoke()
        n_infer += 1
        last = interp.get_tensor(output_index)[0]
        prob = float(softmax(last)[SIRI_INDEX])
        if prob > max_prob:
            max_prob = prob
        if prob >= wake_threshold:
            hits += 1
            if hits >= wake_hits:
                woke = True
                break
        else:
            hits = 0
    return woke, max_prob, last, n_infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/e/kwsDataSet/produce")
    parser.add_argument(
        "--tflite",
        default="/mnt/e/WorkSpace/Person/hey_siri_kws/model_mfcc/stream_state_internal_int8.tflite",
    )
    parser.add_argument(
        "--model-dir",
        default="/home/wdf/kws_work/models/siri_bc_resnet2_siri4500",
    )
    parser.add_argument("--backend", choices=("keras", "tflite"), default="tflite")
    parser.add_argument("--mode", choices=("clip", "wake"), default="wake",
                        help="clip: one 1s forward. wake: C++ sliding-window debounce.")
    parser.add_argument("--split", choices=("all", "training", "validation", "testing"),
                        default="testing")
    parser.add_argument("--keyword", default="siri")
    parser.add_argument("--max-per-label", type=int, default=0,
                        help="Cap negative labels. 0 = all files.")
    parser.add_argument("--max-keyword", type=int, default=0,
                        help="Optional cap on keyword clips. 0 = all keyword files.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-samples", type=int, default=16000)
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--window-stride", type=int, default=320)
    parser.add_argument("--mel-lo", type=float, default=20.0)
    parser.add_argument("--mel-hi", type=float, default=7600.0)
    parser.add_argument("--mel-bins", type=int, default=40)
    parser.add_argument("--dct-bins", type=int, default=20)
    parser.add_argument("--infer-stride", type=int, default=4)
    parser.add_argument("--wake-threshold", type=float, default=0.90)
    parser.add_argument("--wake-hits", type=int, default=2)
    parser.add_argument("--tail-frames", type=int, default=8,
                        help="Silence frames appended after each clip (mic stays on).")
    args = parser.parse_args()

    if args.keyword != "siri":
        log("This script currently assumes label index 2 is siri")
        return 1

    keras_model = None
    interp = None
    input_index = output_index = None
    tflite_clip = False
    time_steps = feat_dim = None
    if args.backend == "keras":
        keras_model = load_keras_model(args.model_dir)
        log(f"backend=keras weights={args.model_dir}/best_weights")
    else:
        interp = tf.lite.Interpreter(model_path=args.tflite)
        interp.allocate_tensors()
        input_index = interp.get_input_details()[0]["index"]
        output_index = interp.get_output_details()[0]["index"]
        in_shape = interp.get_input_details()[0]["shape"]
        tflite_clip = len(in_shape) >= 3 and int(in_shape[1]) > 1
        time_steps = int(in_shape[1])
        feat_dim = int(in_shape[2]) if len(in_shape) >= 3 else int(in_shape[-1])
        log(f"backend=tflite model={args.tflite}")
        log(f"input={in_shape} output={interp.get_output_details()[0]['shape']} clip={tflite_clip}")

    if args.mode == "wake" and args.backend == "tflite" and not tflite_clip:
        log("ERROR: wake mode needs a clip TFLite input [1, 50, 20]")
        return 1

    dataset = list_dataset(
        args.data_dir, args.keyword, args.max_per_label, args.max_keyword, args.split)
    total_files = sum(len(wavs) for _, wavs in dataset)
    log(f"split={args.split} mode={args.mode} labels={len(dataset)} files={total_files}")
    if args.mode == "wake":
        log(f"wake: stride={args.infer_stride} thr={args.wake_threshold} "
            f"hits={args.wake_hits} tail_frames={args.tail_frames}")

    stats = {}
    done = 0
    skipped = 0
    for label, wavs in dataset:
        wakes = 0
        hits_last = 0
        scores = []
        probs = []
        for path in wavs:
            done += 1
            try:
                if args.mode == "wake":
                    pcm = read_wav_pcm16(
                        path, args.sample_rate, min_samples=args.clip_samples, truncate=False)
                    if args.tail_frames > 0:
                        pcm = np.pad(pcm, (0, args.tail_frames * args.window_stride))
                else:
                    pcm = read_wav_pcm16(
                        path, args.sample_rate, min_samples=args.clip_samples, truncate=True)
                mfcc = pcm_to_mfcc(
                    pcm,
                    args.sample_rate,
                    args.window_size,
                    args.window_stride,
                    args.mel_lo,
                    args.mel_hi,
                    args.mel_bins,
                    args.dct_bins,
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                log(f"SKIP {path}: {exc}")
                continue

            if args.mode == "wake":
                woke, max_prob, last, _n_infer = run_clip_wake(
                    interp, input_index, output_index, mfcc, time_steps, feat_dim,
                    args.infer_stride, args.wake_threshold, args.wake_hits)
                if woke:
                    wakes += 1
                probs.append(max_prob)
                scores.append(float(last[SIRI_INDEX]) if last is not None else 0.0)
            elif args.backend == "keras":
                last = keras_model(mfcc[np.newaxis, ...], training=False).numpy()[0]
                prob = float(softmax(last)[SIRI_INDEX])
                if int(np.argmax(last)) == SIRI_INDEX:
                    hits_last += 1
                probs.append(prob)
                scores.append(float(last[SIRI_INDEX]))
            else:
                if tflite_clip:
                    last, max_siri, max_prob = run_clip_tflite(
                        interp, input_index, output_index, mfcc, time_steps, feat_dim)
                else:
                    last, max_siri, max_prob = run_stream(
                        interp, input_index, output_index, mfcc)
                if int(np.argmax(last)) == SIRI_INDEX:
                    hits_last += 1
                probs.append(max_prob)
                scores.append(max_siri)

            if done % 200 == 0:
                log(f"progress {done}/{total_files}")

        n = len(probs)
        stats[label] = {
            "n": n,
            "wakes": wakes,
            "hit_last": hits_last,
            "mean_siri_prob": float(np.mean(probs)) if n else 0.0,
            "probs": probs,
        }
        if n:
            hit = wakes if args.mode == "wake" else hits_last
            log(f"done {label}: {hit}/{n} = {hit / n:.2%}")

    pos = stats.get(args.keyword)
    if not pos or pos["n"] == 0:
        log(f"ERROR: no usable {args.keyword} clips")
        return 1

    hit_key = "wakes" if args.mode == "wake" else "hit_last"
    title = (
        f"=== realtime wake (stride={args.infer_stride} thr={args.wake_threshold} "
        f"hits={args.wake_hits}) ==="
        if args.mode == "wake"
        else "=== clip classification (argmax) ==="
    )
    log("")
    log(title)
    log(f"{'label':<16} {'n':>6} {'wake':>8} {'rate':>8} {'mean_p':>8}")
    neg_n = 0
    neg_fa = 0
    for label, st in stats.items():
        hit = st[hit_key]
        rate = hit / st["n"] if st["n"] else 0.0
        log(f"{label:<16} {st['n']:6d} {hit:8d} {rate:8.2%} {st['mean_siri_prob']:8.3f}")
        if label != args.keyword:
            neg_n += st["n"]
            neg_fa += hit

    recall = pos[hit_key] / pos["n"]
    far = neg_fa / neg_n if neg_n else 0.0
    log("")
    log(f"siri wake/recall = {recall:.2%}  ({pos[hit_key]}/{pos['n']})")
    log(f"false accept     = {far:.2%}  ({neg_fa}/{neg_n})")

    dangerous = [
        (lab, st) for lab, st in stats.items()
        if lab != args.keyword and st["n"] and st[hit_key] / st["n"] >= 0.05
    ]
    dangerous.sort(key=lambda kv: kv[1][hit_key] / kv[1]["n"], reverse=True)
    if dangerous:
        log("")
        log("=== high false-accept labels (>=5%) ===")
        for lab, st in dangerous[:20]:
            log(f"  {lab:<16} {st[hit_key]}/{st['n']} = {st[hit_key]/st['n']:.2%}")

    log(f"\nskipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
