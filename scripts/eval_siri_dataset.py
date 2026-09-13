#!/usr/bin/env python3
"""Evaluate streaming INT8 KWS model on a folder-per-label WAV dataset."""

import argparse
import os
import sys
import wave

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops


LABELS = ("_silence_", "_unknown_", "siri")
SIRI_INDEX = 2


def log(msg):
    print(msg, flush=True)


def read_wav_pcm16(path, sample_rate=16000, desired_samples=16000):
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
    if pcm.size < desired_samples:
        pcm = np.pad(pcm, (0, desired_samples - pcm.size))
    elif pcm.size > desired_samples:
        pcm = pcm[:desired_samples]
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


def list_dataset(data_dir, keyword, max_per_label, max_keyword=0):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/e/kwsDataSet/produce")
    parser.add_argument(
        "--tflite",
        default="/mnt/e/WorkSpace/Person/hey_siri_kws/model_mfcc/stream_state_internal_int8.tflite",
    )
    parser.add_argument(
        "--model-dir",
        default="/home/wdf/kws_work/models/siri_bc_resnet2_run",
    )
    parser.add_argument("--backend", choices=("keras", "tflite"), default="keras")
    parser.add_argument("--keyword", default="siri")
    parser.add_argument("--max-per-label", type=int, default=80,
                        help="Cap negative labels; keyword folder is never capped unless --max-keyword. 0 = all files.")
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
    args = parser.parse_args()

    if args.keyword != "siri":
        log("This script currently assumes label index 2 is siri")
        return 1

    keras_model = None
    interp = None
    input_index = output_index = None
    tflite_clip = False
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
        log(f"backend=tflite model={args.tflite}")
        log(f"input={in_shape} output={interp.get_output_details()[0]['shape']} clip={tflite_clip}")

    dataset = list_dataset(args.data_dir, args.keyword, args.max_per_label, args.max_keyword)
    total_files = sum(len(wavs) for _, wavs in dataset)
    log(f"labels={len(dataset)} files={total_files} max_per_label={args.max_per_label}")

    stats = {}
    done = 0
    skipped = 0
    for label, wavs in dataset:
        hits_last = 0
        hits_max = 0
        scores = []
        probs = []
        feats = []
        for path in wavs:
            done += 1
            try:
                pcm = read_wav_pcm16(path, args.sample_rate, args.clip_samples)
                feats.append(
                    pcm_to_mfcc(
                        pcm,
                        args.sample_rate,
                        args.window_size,
                        args.window_stride,
                        args.mel_lo,
                        args.mel_hi,
                        args.mel_bins,
                        args.dct_bins,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                log(f"SKIP {path}: {exc}")
            if done % 1000 == 0:
                log(f"progress {done}/{total_files}")
        results = []
        if args.backend == "keras":
            for feat in feats:
                last = keras_model(feat[np.newaxis, ...], training=False).numpy()[0]
                prob = float(softmax(last)[SIRI_INDEX])
                results.append((last, float(last[SIRI_INDEX]), prob))
        else:
            in_shape = interp.get_input_details()[0]["shape"]
            for feat in feats:
                if tflite_clip:
                    results.append(
                        run_clip_tflite(
                            interp,
                            input_index,
                            output_index,
                            feat,
                            int(in_shape[1]),
                            int(in_shape[2]),
                        )
                    )
                else:
                    results.append(run_stream(interp, input_index, output_index, feat))
        for last, max_siri, max_prob in results:
            pred_last = int(np.argmax(last))
            if pred_last == SIRI_INDEX or (
                max_siri > float(np.max(np.delete(last, SIRI_INDEX)))
            ):
                pred_max = SIRI_INDEX
            else:
                pred_max = pred_last
            if pred_last == SIRI_INDEX:
                hits_last += 1
            if pred_max == SIRI_INDEX:
                hits_max += 1
            scores.append(max_siri)
            probs.append(max_prob)
        n = len(scores)
        stats[label] = {
            "n": n,
            "hit_last": hits_last,
            "hit_max": hits_max,
            "mean_siri_logit": float(np.mean(scores)) if n else 0.0,
            "mean_siri_prob": float(np.mean(probs)) if n else 0.0,
            "probs": probs,
        }

    pos = stats.get(args.keyword)
    if not pos or pos["n"] == 0:
        log(f"ERROR: no usable {args.keyword} clips")
        return 1

    log("")
    log("=== clip classification (last frame argmax) ===")
    log(f"{'label':<16} {'n':>6} {'pred_siri':>10} {'rate':>8} {'mean_p':>8}")
    neg_n = 0
    neg_fa = 0
    for label, st in stats.items():
        rate = st["hit_last"] / st["n"] if st["n"] else 0.0
        log(f"{label:<16} {st['n']:6d} {st['hit_last']:10d} {rate:8.2%} {st['mean_siri_prob']:8.3f}")
        if label != args.keyword:
            neg_n += st["n"]
            neg_fa += st["hit_last"]

    recall = pos["hit_last"] / pos["n"]
    far = neg_fa / neg_n if neg_n else 0.0
    log("")
    log(f"siri recall (last-frame argmax) = {recall:.2%}  ({pos['hit_last']}/{pos['n']})")
    log(f"false accept (other words)      = {far:.2%}  ({neg_fa}/{neg_n})")

    log("")
    log("=== threshold sweep on max-over-time siri softmax ===")
    pos_p = np.array(pos["probs"], dtype=np.float32)
    neg_p = np.concatenate(
        [np.array(st["probs"], dtype=np.float32) for lab, st in stats.items() if lab != args.keyword]
    ) if neg_n else np.array([], dtype=np.float32)
    log(f"{'thr':>6} {'recall':>8} {'FAR':>8} {'TP':>6} {'FN':>6} {'FP':>6} {'TN':>6}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        tp = int(np.sum(pos_p >= thr))
        fn = pos["n"] - tp
        fp = int(np.sum(neg_p >= thr)) if neg_p.size else 0
        tn = neg_n - fp
        log(f"{thr:6.2f} {tp / pos['n']:8.2%} {fp / neg_n if neg_n else 0:8.2%} {tp:6d} {fn:6d} {fp:6d} {tn:6d}")

    dangerous = [
        (lab, st) for lab, st in stats.items()
        if lab != args.keyword and st["n"] and st["hit_last"] / st["n"] >= 0.05
    ]
    dangerous.sort(key=lambda kv: kv[1]["hit_last"] / kv[1]["n"], reverse=True)
    if dangerous:
        log("")
        log("=== high false-accept labels (>=5%) ===")
        for lab, st in dangerous[:20]:
            log(f"  {lab:<16} {st['hit_last']}/{st['n']} = {st['hit_last']/st['n']:.2%}")

    log(f"\nskipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
