#!/usr/bin/env bash
# P1: B3 combo D32 | P2: тот же combo, PARITY_EPOCHS=96 | P3: longcontext 96,512, PARITY_BATCH=8
# Запуск: cd ~/Desktop/WFR-Memory-Test && bash _remote_pipeline_p1_p2_p3.sh
set -eu
ROOT="${WFR_REMOTE_ROOT:-$HOME/Desktop/WFR-Memory-Test}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p experiments/09-lm-parity/outputs experiments/09-lm-parity/outputs/remote_a100/runs

for f in experiments/09-lm-parity/*.sh; do
  [[ -f "$f" ]] || continue
  sed -i 's/\r$//' "$f"
  chmod +x "$f"
done

echo "========== P1 B3 combo D32 (48 ep default) $(date -Is) =========="
bash experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh

echo "========== P2 B3 combo D32 PARITY_EPOCHS=96 $(date -Is) =========="
PARITY_EPOCHS=96 bash experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh

echo "========== P3 longcontext D32 seq 96,512 PARITY_BATCH=8 $(date -Is) =========="
PARITY_BATCH=8 bash experiments/09-lm-parity/remote_gpu_longcontext_d32.sh

echo "========== PIPELINE P1+P2+P3 OK $(date -Is) =========="
