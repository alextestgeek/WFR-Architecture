# Запуск Experiment 06 на удалённом GPU (локальный контур)

Параметры SSH, хост, пароль и host key — в **локальном** `RULES.md` (раздел 6). В репозиторий они не коммитятся.

## Что залить на сервер

Структура под корень `~/Desktop/WFR-Memory-Test/` (как в RULES):

- `experiments/00-smoke-test/wfr_core.py`
- `experiments/00-smoke-test/phase0_best_config.py`
- `wfr_lm.py`, `wfr_rfp.py` (в корень `WFR-Memory-Test`)
- `experiments/06-rfp-v0/run_rfp_training.py`
- `experiments/06-rfp-v0/test_rfp_vs_baseline.py`

Каталоги создать заранее: `experiments/00-smoke-test`, `experiments/06-rfp-v0`, `outputs`.

## PowerShell: пример (подставить `$H` и пароль из RULES)

```powershell
$H = "<hostkey из RULES.md>"
$ROOT = "c:\WFR-Architecture"
$R = "alex1@<HOST>:/home/alex1/Desktop/WFR-Memory-Test"

& "C:\Program Files\PuTTY\plink.exe" -ssh -batch -pw "<PASSWORD>" -hostkey $H alex1@<HOST> `
  "mkdir -p ~/Desktop/WFR-Memory-Test/experiments/00-smoke-test ~/Desktop/WFR-Memory-Test/experiments/06-rfp-v0 ~/Desktop/WFR-Memory-Test/outputs"

& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H "$ROOT\experiments\00-smoke-test\wfr_core.py" "$R/experiments/00-smoke-test/wfr_core.py"
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H "$ROOT\experiments\00-smoke-test\phase0_best_config.py" "$R/experiments/00-smoke-test/phase0_best_config.py"
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H "$ROOT\wfr_lm.py" "$R/wfr_lm.py"
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H "$ROOT\wfr_rfp.py" "$R/wfr_rfp.py"
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H "$ROOT\experiments\06-rfp-v0\run_rfp_training.py" "$R/experiments/06-rfp-v0/run_rfp_training.py"
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H "$ROOT\experiments\06-rfp-v0\test_rfp_vs_baseline.py" "$R/experiments/06-rfp-v0/test_rfp_vs_baseline.py"
```

## Запуск на сервере

Только интерпретатор из venv (PyTorch CUDA), из корня `WFR-Memory-Test`:

```powershell
$remote = "cd ~/Desktop/WFR-Memory-Test && mkdir -p outputs && .venv/bin/python experiments/06-rfp-v0/test_rfp_vs_baseline.py --rfp-version v02 --epochs 50 2>&1 | tee outputs/exp06_rfp_v02_remote.log"
& "C:\Program Files\PuTTY\plink.exe" -ssh -batch -pw "<PASSWORD>" -hostkey $H alex1@<HOST> $remote
```

Другие режимы: `--rfp-version all`, `grid_rfp_v02.py` — при необходимости также скопировать `grid_rfp_v02.py` в `experiments/06-rfp-v0/`.

## Скачать результаты

```powershell
$OUT = "c:\WFR-Architecture\experiments\06-rfp-v0\outputs\remote_gpu"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H `
  "alex1@<HOST>:/home/alex1/Desktop/WFR-Memory-Test/experiments/06-rfp-v0/outputs/ab_rfp_v02.json" "$OUT\"
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H `
  "alex1@<HOST>:/home/alex1/Desktop/WFR-Memory-Test/experiments/06-rfp-v0/outputs/rfp_v02_log.json" "$OUT\"
& "C:\Program Files\PuTTY\pscp.exe" -pw "<PASSWORD>" -hostkey $H `
  "alex1@<HOST>:/home/alex1/Desktop/WFR-Memory-Test/outputs/exp06_rfp_v02_remote.log" "$OUT\"
```

Локальная копия последнего прогона: каталог `outputs/remote_gpu/`.
