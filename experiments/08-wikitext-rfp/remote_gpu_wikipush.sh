#!/usr/bin/env bash
# Запуск wiki-push на удалённой машине (из ~/Desktop/WFR-Memory-Test, см. RULES.md §6).
# Использует python -u; лог с меткой времени, чтобы не затирать предыдущие прогоны.
set -euo pipefail
ROOT="${WFR_REMOTE_ROOT:-$HOME/Desktop/WFR-Memory-Test}"
cd "$ROOT"
mkdir -p experiments/08-wikitext-rfp/outputs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="experiments/08-wikitext-rfp/outputs/wikipush_${TS}.log"
echo "Logging to $LOG"
nohup .venv/bin/python -u experiments/08-wikitext-rfp/run_wikitext_train.py \
  --wiki-push --rfp-version v03 \
  >"$LOG" 2>&1 &
echo "PID $! — tail -f $LOG"
echo "Ожидание завершения (безопасно для pgrep): bash experiments/08-wikitext-rfp/wait_wikitext_python.sh"
