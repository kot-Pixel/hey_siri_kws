#!/bin/bash
# Train hey_siri MFCC model and export INT8 streaming TFLite.
set -euo pipefail

source /home/wdf/miniconda3/etc/profile.d/conda.sh
conda activate kws39
export PYTHONPATH=/home/wdf/kws_work/google-research
cd /home/wdf/kws_work/google-research

TRAIN_DIR=/home/wdf/kws_work/models/hey_siri_mfcc_int8_run
DATA_DIR=/home/wdf/kws_work/data/content/drive/MyDrive/kws_data/kws_train_data

rm -rf "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=-1 python -m kws_streaming.train.model_train_eval \
  --data_url '' \
  --data_dir "$DATA_DIR" \
  --train_dir "$TRAIN_DIR" \
  --wanted_words hey_siri \
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
  --silence_percentage 10 \
  --unknown_percentage 60 \
  --split_data 1 \
  --train 1 \
  --how_many_training_steps 10000,10000,10000 \
  --eval_step_interval 500 \
  --save_step_interval 1000 \
  --learning_rate 0.001,0.0005,0.0001 \
  --alsologtostderr \
  svdf \
  --svdf_memory_size '4,10,10' \
  --svdf_units1 '64,64,64' \
  --svdf_units2 '32,32,-1' \
  --svdf_act "'relu','relu','relu'" \
  --svdf_dropout '0.0,0.0,0.0' \
  --svdf_pad 1 \
  --dropout1 0.0

python /home/wdf/kws_work/convert_streaming_int8.py \
  --model_dir "$TRAIN_DIR" \
  --rep_samples 200

echo "INT8 model:"
ls -lh "$TRAIN_DIR/tflite_stream_state_internal/stream_state_internal_int8.tflite"
