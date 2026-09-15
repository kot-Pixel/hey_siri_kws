#!/bin/bash
# Train siri BC-ResNet-2 MFCC model and export INT8 streaming TFLite.
# Data: /mnt/e/kwsDataSet/produce (Windows E:\kwsDataSet\produce)
# Preprocessing is identical to SVDF: mfcc, 20ms window/stride, 40 mel, 20 DCT.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

source /home/wdf/miniconda3/etc/profile.d/conda.sh
conda activate kws39
export PYTHONPATH=/home/wdf/kws_work/google-research
cd /home/wdf/kws_work/google-research

TRAIN_DIR=/home/wdf/kws_work/models/siri_bc_resnet2_siri4500
SRC_DATA=/mnt/e/kwsDataSet/produce
DATA_DIR=/home/wdf/kws_work/data/siri_hardneg

# 混淆词全部进入 unknown；剩余 0.7*|siri| 名额从其它关键字目录抽。
python "$PROJECT_ROOT/scripts/make_unknown_quota_datadir.py" \
  --src "$SRC_DATA" \
  --dst "$DATA_DIR" \
  --unknown-percentage 70

rm -rf "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=-1 python -m kws_streaming.train.model_train_eval \
  --data_url '' \
  --data_dir "$DATA_DIR" \
  --train_dir "$TRAIN_DIR" \
  --wanted_words siri \
  --sample_rate 16000 \
  --clip_duration_ms 1000 \
  --window_size_ms 20.0 \
  --window_stride_ms 20.0 \
  --preprocess mfcc \
  --mel_upper_edge_hertz 7600 \
  --mel_lower_edge_hertz 20 \
  --mel_num_bins 40 \
  --dct_num_features 20 \
  --quantize 1 \
  --resample 0.15 \
  --background_frequency 0.8 \
  --background_volume 0.1 \
  --time_shift_ms 100 \
  --silence_percentage 15 \
  --unknown_percentage 70 \
  --split_data 1 \
  --train 1 \
  --how_many_training_steps 30000,30000,30000 \
  --eval_step_interval 1000 \
  --save_step_interval 2000 \
  --learning_rate 0.001,0.0005,0.0001 \
  --use_spec_augment 1 \
  --time_masks_number 2 \
  --time_mask_max_size 20 \
  --frequency_masks_number 2 \
  --frequency_mask_max_size 3 \
  --alsologtostderr \
  bc_resnet \
  --paddings 'causal' \
  --sub_groups 5 \
  --first_filters 16 \
  --last_filters 32 \
  --blocks_n '2, 2, 4, 4' \
  --filters '16, 24, 32, 40' \
  --dilations '(1,1),(2,1),(4,1),(8,1)' \
  --strides '(1,1),(1,2),(1,1),(1,1)' \
  --dropouts '0.1, 0.1, 0.1, 0.1' \
  --pools '1, 1, 1, 1' \
  --max_pool 0

python "$PROJECT_ROOT/scripts/convert_streaming_int8.py" \
  --model_dir "$TRAIN_DIR" \
  --clip \
  --data_dir "$SRC_DATA" \
  --rep_samples 400

echo "INT8 model:"
ls -lh "$TRAIN_DIR/tflite_stream_state_internal/stream_state_internal_int8.tflite"
