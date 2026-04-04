#!/usr/bin/env bash
# A100: fair-parity сетка B1 — readout_feat_dim × readout_mlp_hidden (док. 14 §4 п.2).
# Дерево: ~/Desktop/WFR-Memory-Test, venv .venv (RULES.md §6).
#
# По умолчанию: dims 16 32, MLP hidden 64 (и контроль без MLP для каждого dim).
# Переопределение:
#   B1_READOUT_DIMS="16 32" B1_MLP_HIDDEN_LIST="64" NO_MLP_BASELINE=0 bash remote_gpu_b1_mlp_matrix.sh
#   NO_MLP_BASELINE=1 — только пары с --readout-mlp-hidden (без чистого linear head).
set -eu
ROOT="${WFR_REMOTE_ROOT:-$HOME/Desktop/WFR-Memory-Test}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p experiments/09-lm-parity/outputs

TS="$(date +%Y%m%d_%H%M%S)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv: $PY"
  exit 1
fi

echo "=== GPU check ==="
"$PY" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

EPOCHS="${PARITY_EPOCHS:-48}"
SEQ="${PARITY_SEQ_LEN:-96}"
NTR="${PARITY_NUM_TRAIN:-20}"
NV="${PARITY_NUM_VAL:-8}"
BS="${PARITY_BATCH:-16}"
SKIP_BASELINE="${NO_MLP_BASELINE:-0}"

run_pair() {
  local tag="$1"
  shift
  local log="experiments/09-lm-parity/outputs/b1mlp_${tag}_${TS}.log"
  echo ""
  echo "========== $tag → $log =========="
  "$PY" -u experiments/09-lm-parity/run_parity_pair.py \
    --fair-parity \
    --epochs "$EPOCHS" \
    --seq-len "$SEQ" \
    --num-train-batches "$NTR" \
    --num-val-batches "$NV" \
    --batch-size "$BS" \
    "$@" 2>&1 | tee "$log"
}

# shellcheck disable=SC2206
Dims=( ${B1_READOUT_DIMS:-16 32} )
MlpList=( ${B1_MLP_HIDDEN_LIST:-64} )

for d in "${Dims[@]}"; do
  if [[ "$SKIP_BASELINE" != "1" ]]; then
    if [[ "$d" -eq 3 ]]; then
      run_pair "d${d}_linear"
    else
      run_pair "d${d}_linear" --readout-feat-dim "$d"
    fi
  fi
  for h in "${MlpList[@]}"; do
    if [[ "$d" -eq 3 ]]; then
      run_pair "d${d}_mlp${h}" --readout-mlp-hidden "$h"
    else
      run_pair "d${d}_mlp${h}" --readout-feat-dim "$d" --readout-mlp-hidden "$h"
    fi
  done
done

echo ""
echo "OK B1 MLP matrix TS=$TS — JSON under experiments/09-lm-parity/outputs/"
