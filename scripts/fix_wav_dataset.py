#!/usr/bin/env python3
"""Convert dataset WAV files to PCM s16le mono 16kHz for kws_streaming."""

import argparse
import os
import struct
import subprocess
import sys
import tempfile


def read_wav_fmt(path):
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            return None, "not_riff"
        f.read(4)
        if f.read(4) != b"WAVE":
            return None, "not_wave"

        fmt_code = None
        channels = None
        sample_rate = None
        bits = None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            (chunk_size,) = struct.unpack("<I", f.read(4))
            if chunk_id == b"fmt ":
                data = f.read(min(chunk_size, 16))
                if len(data) >= 16:
                    fmt_code, channels = struct.unpack("<HH", data[:4])
                    sample_rate = struct.unpack("<I", data[4:8])[0]
                    bits = struct.unpack("<H", data[14:16])[0]
            else:
                f.seek(chunk_size, 1)
            if fmt_code is not None and chunk_id != b"fmt ":
                break

    if fmt_code is None:
        return None, "no_fmt"
    return {
        "fmt": fmt_code,
        "channels": channels,
        "rate": sample_rate,
        "bits": bits,
    }, None


def needs_convert(info):
    if info is None:
        return True
    return not (
        info["fmt"] == 1
        and info["channels"] == 1
        and info["rate"] == 16000
        and info["bits"] == 16
    )


def convert_with_ffmpeg(src, dst):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-c:a",
        "pcm_s16le",
        dst,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0


def process_file(path, dry_run, verbose):
    info, err = read_wav_fmt(path)
    if err == "not_riff" or err == "not_wave" or err == "no_fmt":
        if verbose:
            print(f"SKIP invalid: {path} ({err})")
        return "skip"

    if not needs_convert(info):
        return "ok"

    reason = []
    if info["fmt"] != 1:
        reason.append(f"fmt={info['fmt']}")
    if info["channels"] != 1:
        reason.append(f"ch={info['channels']}")
    if info["rate"] != 16000:
        reason.append(f"sr={info['rate']}")
    if info["bits"] != 16:
        reason.append(f"bits={info['bits']}")
    if verbose:
        print(f"FIX {' '.join(reason)}: {path}")

    if dry_run:
        return "fix"

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        if not convert_with_ffmpeg(path, tmp_path):
            print(f"FAIL ffmpeg: {path}", file=sys.stderr)
            return "fail"
        os.replace(tmp_path, path)
        return "fixed"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/home/wdf/kws_work/data/content/drive/MyDrive/kws_data/kws_train_data",
    )
    parser.add_argument("--label", default="", help="Only process one label folder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not shutil_which("ffmpeg"):
        print("ERROR: ffmpeg not found. Install: sudo apt install -y ffmpeg", file=sys.stderr)
        return 1

    root = args.data_dir
    if args.label:
        root = os.path.join(root, args.label)

    stats = {"ok": 0, "fix": 0, "fixed": 0, "fail": 0, "skip": 0}
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if not name.lower().endswith(".wav"):
                continue
            result = process_file(os.path.join(dirpath, name), args.dry_run, args.verbose)
            stats[result] = stats.get(result, 0) + 1

    print(
        "done:",
        f"ok={stats.get('ok', 0)}",
        f"would_fix={stats.get('fix', 0)}",
        f"fixed={stats.get('fixed', 0)}",
        f"fail={stats.get('fail', 0)}",
        f"skip={stats.get('skip', 0)}",
    )
    return 1 if stats.get("fail", 0) else 0


def shutil_which(cmd):
    from shutil import which

    return which(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
