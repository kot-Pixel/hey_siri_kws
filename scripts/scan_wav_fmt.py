#!/usr/bin/env python3
"""Quick scan for non-PCM WAV files (fmt != 1)."""

import argparse
import os
import struct
from collections import Counter


def wav_fmt(path):
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            return None
        f.read(4)
        if f.read(4) != b"WAVE":
            return None
        while True:
            cid = f.read(4)
            if len(cid) < 4:
                return None
            (sz,) = struct.unpack("<I", f.read(4))
            if cid == b"fmt ":
                data = f.read(min(sz, 2))
                if len(data) < 2:
                    return None
                return struct.unpack("<H", data)[0]
            f.seek(sz, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/home/wdf/kws_work/data/content/drive/MyDrive/kws_data/kws_train_data",
    )
    args = parser.parse_args()

    bad = Counter()
    examples = []
    total = 0
    for dirpath, _, files in os.walk(args.data_dir):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            total += 1
            path = os.path.join(dirpath, name)
            fmt = wav_fmt(path)
            if fmt != 1:
                label = os.path.relpath(path, args.data_dir).split(os.sep)[0]
                bad[(label, fmt)] += 1
                if len(examples) < 15:
                    examples.append((fmt, path))

    print(f"total_wav={total}")
    print(f"bad_count={sum(bad.values())}")
    for (label, fmt), count in bad.most_common(20):
        print(f"  {label}: fmt={fmt} count={count}")
    for fmt, path in examples:
        print(f"example fmt={fmt}: {path}")


if __name__ == "__main__":
    main()
