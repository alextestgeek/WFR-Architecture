#!/usr/bin/env bash
# A100: B3 combo — readout_wave_kernel=3 + MLP 64 при readout_feat_dim=32 (fair-parity, как B1 бюджет).
# Из ~/Desktop/WFR-Memory-Test после upload.
set -eu
ROOT="${WFR_REMOTE_ROOT:-$HOME/Desktop/WFR-Memory-Test}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p experiments/09-lm-parity/outputs
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv: $PY"
  exit 1
fi
echo "=== GPU ==="
"$PY" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

EPOCHS="${PARITY_EPOCHS:-48}"
SEQ="${PARITY_SEQ_LEN:-96}"
NTR="${PARITY_NUM_TRAIN:-20}"
NV="${PARITY_NUM_VAL:-8}"
BS="${PARITY_BATCH:-16}"

LOG="experiments/09-lm-parity/outputs/b3_combo_d32_rw3_mlp64_$(date +%Y%m%d_%H%M%S).log"
echo "========== B3 combo D32 rw3 mlp64 → $LOG =========="
"$PY" -u experiments/09-lm-parity/run_parity_pair.py \
  --fair-parity \
  --epochs "$EPOCHS" \
  --seq-len "$SEQ" \
  --num-train-batches "$NTR" \
  --num-val-batches "$NV" \
  --batch-size "$BS" \
  --readout-feat-dim 32 \
  --readout-mlp-hidden 64 \
  --readout-wave-kernel 3 \
  2>&1 | tee "$LOG"

echo OK
