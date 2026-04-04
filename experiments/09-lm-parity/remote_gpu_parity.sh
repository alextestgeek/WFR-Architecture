#!/usr/bin/env bash
# Длинный парный прогон A2 на A100 (RULES.md §6): из ~/Desktop/WFR-Memory-Test.
# Серия прогонов B1: набор readout_feat_dim (по умолчанию 3 и 16).
# Несколько значений: PARITY_READOUT_DIMS="3 8 16 32" bash remote_gpu_parity.sh
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

# Бюджет ближе к Exp 08 --long-run (не wiki-push): честное сравнение двух моделей ≈ 2× время одного long-run
EPOCHS="${PARITY_EPOCHS:-48}"
SEQ="${PARITY_SEQ_LEN:-96}"
NTR="${PARITY_NUM_TRAIN:-20}"
NV="${PARITY_NUM_VAL:-8}"
BS="${PARITY_BATCH:-16}"

# Опционально: PARITY_READOUT_MLP_HIDDEN=64 — MLP readout (см. wfr_lm.WFRLM, дорожная карта B1)
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

run_pair() {
  local tag="$1"
  shift
  local log="experiments/09-lm-parity/outputs/parity_a100_${tag}_${TS}.log"
  echo ""
  echo "========== $tag → $log =========="
  "$PY" -u experiments/09-lm-parity/run_parity_pair.py \
    --fair-parity \
    --epochs "$EPOCHS" \
    --seq-len "$SEQ" \
    --num-train-batches "$NTR" \
    --num-val-batches "$NV" \
    --batch-size "$BS" \
    "${MLP_ARG[@]}" \
    "${NB_ARG[@]}" \
    "${PHASE_ARG[@]}" \
    "${RW_ARG[@]}" \
    "$@" 2>&1 | tee "$log"
}

# shellcheck disable=SC2086
Dims="${PARITY_READOUT_DIMS:-3 16}"
for d in $Dims; do
  mlp_suffix=""
  if [[ -n "${PARITY_READOUT_MLP_HIDDEN:-}" ]]; then
    mlp_suffix="_mlp${PARITY_READOUT_MLP_HIDDEN}"
  fi
  if [[ "${PARITY_CONTENT_NEIGHBOR_MIX:-}" == "1" ]]; then
    mlp_suffix="${mlp_suffix}_nb"
  fi
  if [[ -n "${PARITY_PHASE_CAUSAL_KERNEL:-}" ]] && [[ "${PARITY_PHASE_CAUSAL_KERNEL}" -gt 1 ]]; then
    mlp_suffix="${mlp_suffix}_pk${PARITY_PHASE_CAUSAL_KERNEL}"
  fi
  if [[ -n "${PARITY_READOUT_WAVE_KERNEL:-}" ]] && [[ "${PARITY_READOUT_WAVE_KERNEL}" -gt 1 ]]; then
    mlp_suffix="${mlp_suffix}_rw${PARITY_READOUT_WAVE_KERNEL}"
  fi
  if [ "$d" -eq 3 ] 2>/dev/null; then
    run_pair "readout${d}${mlp_suffix}"
  else
    run_pair "readout${d}${mlp_suffix}" --readout-feat-dim "$d"
  fi
done

echo ""
echo "OK parity A100 batch TS=$TS — JSON under experiments/09-lm-parity/outputs/"
