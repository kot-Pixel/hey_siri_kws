#!/usr/bin/env python3
"""Build a data_dir so high-confusion words are always in the unknown slice.

kws_streaming keeps only unknown_percentage * |siri| unknown files per split,
then shuffle-slices the rest away. This writes a curated tree:

  - siri/: all positive wavs
  - each confusion folder: ALL wavs
  - other keyword folders: just enough wavs to fill the 0.7 * siri quota

Use the same filename hash as kws_streaming so the later train split matches.
If curated unknown count <= quota, every file is trained — confusion included.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import shutil
from collections import defaultdict

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
KEYWORD = "siri"


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


def list_wavs(folder):
  if not os.path.isdir(folder):
    return []
  return [
      os.path.join(folder, name)
      for name in sorted(os.listdir(folder))
      if name.lower().endswith(".wav")
  ]


def load_confusion(path, src_root):
  names = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      name = line.strip().lower()
      if not name or name.startswith("#"):
        continue
      names.append(name)
  present = [n for n in names if os.path.isdir(os.path.join(src_root, n))]
  missing = [n for n in names if n not in present]
  return present, missing


def group_by_set(paths):
  grouped = defaultdict(list)
  for path in paths:
    grouped[which_set(path)].append(path)
  return grouped


def symlink_file(src, dst_dir):
  os.makedirs(dst_dir, exist_ok=True)
  dst = os.path.join(dst_dir, os.path.basename(src))
  if os.path.lexists(dst):
    return
  os.symlink(src, dst)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--src", default="/mnt/e/kwsDataSet/produce")
  parser.add_argument("--dst", default="/home/wdf/kws_work/data/siri_hardneg")
  parser.add_argument(
      "--confusion-list",
      default=os.path.join(os.path.dirname(__file__), "confusion_words.txt"),
  )
  parser.add_argument("--keyword", default=KEYWORD)
  parser.add_argument("--unknown-percentage", type=float, default=70.0)
  parser.add_argument("--seed", type=int, default=59185)
  args = parser.parse_args()

  rng = random.Random(args.seed)
  confusion_names, missing = load_confusion(args.confusion_list, args.src)
  if missing:
    print("missing confusion folders (skipped):", ", ".join(missing))

  siri_wavs = list_wavs(os.path.join(args.src, args.keyword))
  if not siri_wavs:
    raise SystemExit(f"no {args.keyword} wavs in {args.src}")
  siri_by_set = group_by_set(siri_wavs)

  confusion_wavs = []
  for name in confusion_names:
    confusion_wavs.extend(list_wavs(os.path.join(args.src, name)))
  confusion_by_set = group_by_set(confusion_wavs)

  other_by_label = {}
  for name in sorted(os.listdir(args.src)):
    path = os.path.join(args.src, name)
    if (
        not os.path.isdir(path)
        or name.startswith("_")
        or name.lower() == args.keyword
        or name.lower() in confusion_names
    ):
      continue
    wavs = list_wavs(path)
    if wavs:
      other_by_label[name] = wavs

  other_pool_by_set = defaultdict(list)
  for label, wavs in other_by_label.items():
    for path in wavs:
      other_pool_by_set[which_set(path)].append((label, path))
  for split in other_pool_by_set:
    rng.shuffle(other_pool_by_set[split])

  selected_others = []
  print("quota per split (unknown_percentage="
        f"{args.unknown_percentage:g}%):")
  for split in ("training", "validation", "testing"):
    siri_n = len(siri_by_set[split])
    budget = int(math.ceil(siri_n * args.unknown_percentage / 100.0))
    conf_n = len(confusion_by_set[split])
    remain = max(0, budget - conf_n)
    picked = other_pool_by_set[split][:remain]
    selected_others.extend(picked)
    print(
        f"  {split:12s} siri={siri_n:5d}  budget={budget:5d}  "
        f"confusion={conf_n:5d}  others={len(picked):5d}  "
        f"{'OK all confusion included' if conf_n <= budget else 'OVERFLOW'}"
    )
    if conf_n > budget:
      raise SystemExit(
          f"{split}: confusion wavs {conf_n} exceed unknown budget {budget}. "
          "Lower confusion count or raise --unknown-percentage."
      )

  if os.path.exists(args.dst):
    shutil.rmtree(args.dst)
  os.makedirs(args.dst)

  bg_src = os.path.join(args.src, "_background_noise_")
  if os.path.isdir(bg_src):
    os.symlink(bg_src, os.path.join(args.dst, "_background_noise_"))

  for path in siri_wavs:
    symlink_file(path, os.path.join(args.dst, args.keyword))
  for path in confusion_wavs:
    symlink_file(path, os.path.join(args.dst, os.path.basename(os.path.dirname(path))))
  for label, path in selected_others:
    symlink_file(path, os.path.join(args.dst, label))

  print(
      f"wrote {args.dst}\n"
      f"  siri={len(siri_wavs)}  confusion_words={len(confusion_names)}  "
      f"confusion_wavs={len(confusion_wavs)}  other_wavs={len(selected_others)}"
  )


if __name__ == "__main__":
  main()
