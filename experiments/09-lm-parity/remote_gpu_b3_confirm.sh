#!/usr/bin/env bash
# Подтверждение на A100 гипотез B3 из локального quick-smoke (phase_causal / readout_wave),
# тот же бюджет шагов, что remote_gpu_parity.sh (RULES.md §6; ~/Desktop/WFR-Memory-Test).
#
# Запуск с хоста:
#   cd ~/Desktop/WFR-Memory-Test && chmod +x experiments/09-lm-parity/remote_gpu_b3_confirm.sh
#   bash experiments/09-lm-parity/remote_gpu_b3_confirm.sh
#
# Переменные (как у parity batch):
#   PARITY_EPOCHS PARITY_SEQ_LEN PARITY_NUM_TRAIN PARITY_NUM_VAL PARITY_BATCH
#   B3_READOUT_DIM — ширина readout WFRLM (по умолчанию 16; узкий 3 даёт другой масштаб CE)
#   B3_INIT_SEED B3_TRAIN_SEED — сиды (по умолчанию 42 / 42)
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
RD="${B3_READOUT_DIM:-16}"
IS="${B3_INIT_SEED:-42}"
TR="${B3_TRAIN_SEED:-42}"

common=(
  --fair-parity
  --epochs "$EPOCHS"
  --seq-len "$SEQ"
  --num-train-batches "$NTR"
  --num-val-batches "$NV"
  --batch-size "$BS"
  --readout-feat-dim "$RD"
  --init-seed "$IS"
  --train-seed "$TR"
)

run_one() {
  local tag="$1"
  shift
  local log="experiments/09-lm-parity/outputs/b3_confirm_${tag}_${TS}.log"
  echo ""
  echo "========== $tag → $log =========="
  "$PY" -u experiments/09-lm-parity/run_parity_pair.py "${common[@]}" "$@" 2>&1 | tee "$log"
}

# 1) Контроль: без локальных флагов B3 (только readout RD)
run_one "control_readout${RD}"

# 2) Каузальная смесь фаз в ядре
run_one "phase_causal_k3" --phase-causal-kernel 3

# 3) Каузальная смесь standing wave перед readout
run_one "readout_wave_k3" --readout-wave-kernel 3

echo ""
echo "OK B3 confirm batch TS=$TS — parity JSON in experiments/09-lm-parity/outputs/"
echo "Сводка: $PY -u experiments/09-lm-parity/summarize_parity_json.py experiments/09-lm-parity/outputs"
