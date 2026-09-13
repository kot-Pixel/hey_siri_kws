#!/bin/bash
set -euo pipefail
source /home/wdf/miniconda3/etc/profile.d/conda.sh
conda activate kws39
export PYTHONPATH=/home/wdf/kws_work/google-research
python /mnt/e/WorkSpace/Person/hey_siri_kws/scripts/eval_siri_dataset.py "$@"
