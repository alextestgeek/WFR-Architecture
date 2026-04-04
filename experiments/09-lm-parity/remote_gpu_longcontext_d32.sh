#!/usr/bin/env bash
# A100 (RULES.md §6): этап D.1 — несколько seq_len подряд, fair-parity, readout по умолчанию 32.
# Из ~/Desktop/WFR-Memory-Test после upload. Один JSON: longcontext_pair_<ts>.json + d1_metrics (wall, peak CUDA).
#
# По умолчанию: seq 96,512; 48 ep; те же батчи, что remote_gpu_parity.sh.
# При OOM на длинной длине: PARITY_BATCH=8 bash remote_gpu_longcontext_d32.sh
# Другие длины: LONGCONTEXT_SEQ_LENS="96,256,512" bash ...
# Другой readout: PARITY_READOUT_DIM=16 bash ...
set -eu
ROOT="${WFR_REMOTE_ROOT:-$HOME/Desktop/WFR-Memory-Test}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p experiments/09-lm-parity/outputs/remote_a100/runs

TS="$(date +%Y%m%d_%H%M%S)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv: $PY"
  exit 1
fi

echo "=== GPU check ==="
"$PY" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

EPOCHS="${PARITY_EPOCHS:-48}"
NTR="${PARITY_NUM_TRAIN:-20}"
NV="${PARITY_NUM_VAL:-8}"
BS="${PARITY_BATCH:-16}"
SEQ_LENS="${LONGCONTEXT_SEQ_LENS:-96,512}"
DIM="${PARITY_READOUT_DIM:-32}"

MLP_ARG=()
if [[ -n "${PARITY_READOUT_MLP_HIDDEN:-}" ]]; then
  MLP_ARG=(--readout-mlp-hidden "${PARITY_READOUT_MLP_HIDDEN}")
fi
NB_ARG=()
if [[ "${PARITY_CONTENT_NEIGHBOR_MIX:-}" == "1" ]]; then
  NB_ARG=(--content-neighbor-mix)
fi
PHASE_ARG=()
if [[ -n "${PARITY_PHASE_CAUSAL_KERNEL:-}" ]] && [[ "${PARITY_PHASE_CAUSAL_KERNEL}" -gt 1 ]]; then
  PHASE_ARG=(--phase-causal-kernel "${PARITY_PHASE_CAUSAL_KERNEL}")
fi
RW_ARG=()
if [[ -n "${PARITY_READOUT_WAVE_KERNEL:-}" ]] && [[ "${PARITY_READOUT_WAVE_KERNEL}" -gt 1 ]]; then
  RW_ARG=(--readout-wave-kernel "${PARITY_READOUT_WAVE_KERNEL}")
fi

RUN_TAG="longcontext_d${DIM}_${TS}"
LOG="experiments/09-lm-parity/outputs/remote_a100/runs/${RUN_TAG}.log"
OUT_DIR="experiments/09-lm-parity/outputs/remote_a100/runs/${RUN_TAG}"
mkdir -p "$OUT_DIR"

echo "========== D.1 longcontext readout=${DIM} seq_lens=${SEQ_LENS} → $LOG =========="
"$PY" -u experiments/09-lm-parity/run_longcontext_pair.py \
  --fair-parity \
  --seq-lens "$SEQ_LENS" \
  --epochs "$EPOCHS" \
  --num-train-batches "$NTR" \
  --num-val-batches "$NV" \
  --batch-size "$BS" \
  --readout-feat-dim "$DIM" \
  "${MLP_ARG[@]}" \
  "${NB_ARG[@]}" \
  "${PHASE_ARG[@]}" \
  "${RW_ARG[@]}" \
  2>&1 | tee "$LOG"

# Копируем свежий JSON рядом с логом (последний по mtime в outputs/)
shopt -s nullglob
LC_FILES=(experiments/09-lm-parity/outputs/longcontext_pair_*.json)
shopt -u nullglob
if [[ ${#LC_FILES[@]} -gt 0 ]]; then
  LATEST="$(ls -t "${LC_FILES[@]}" | head -1)"
  cp -f "$LATEST" "$OUT_DIR/"
  echo "Copied artifact: $OUT_DIR/$(basename "$LATEST")"
fi

echo "OK D.1 batch TS=$TS — see $OUT_DIR and experiments/09-lm-parity/outputs/longcontext_pair_*.json"
