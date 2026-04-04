#!/usr/bin/env bash
# Ждёт завершения процесса Python, который запускает run_wikitext_train.py.
#
# Нельзя использовать в цикле: pgrep -f '.../run_wikitext_train.py' — эта подстрока
# попадает в argv самого bash/plink с тем же pgrep, и цикл никогда не заканчивается.
# Здесь для каждого кандидата из pgrep читается /proc/PID/cmdline и отбираются только
# строки, где есть интерпретатор python (в т.ч. .venv/bin/python).

set -euo pipefail

interval="${WAIT_WIKITEXT_INTERVAL:-60}"

is_python_wikitext_running() {
  local pid cmd
  for pid in $(pgrep -f 'run_wikitext_train\.py' 2>/dev/null || true); do
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmd=$(tr '\0' ' ' <"/proc/${pid}/cmdline")
    # В cmdline обязан быть интерпретатор python; у «битого» bash-only pgrep-цикла слова python нет.
    if echo "$cmd" | grep -qE '(^|[/])python([0-9.]*)?[[:space:]]' && echo "$cmd" | grep -q 'run_wikitext_train'; then
      return 0
    fi
  done
  return 1
}

while is_python_wikitext_running; do
  echo "[$(date -Iseconds)] Python run_wikitext_train still running..."
  sleep "$interval"
done
echo "[$(date -Iseconds)] No Python run_wikitext_train — done."
