# Experiment 06 — protocol tests (fresh data + tiered A/B/C)

Отдельная папка от `06-rfp-v0`: тот же `WFRLM` / RFP, но **протокол данных** ближе к честной проверке сигнала задачи.

## Идея

| Старый toy (`06-rfp-v0`) | Этот протокол |
|--------------------------|----------------|
| Одни и те же train-батчи каждую эпоху | **Новые** train-батчи каждую эпоху (детерминированно от `train_seed + ep`) |
| Val из того же распределения, но смешан с train-логикой | **Фиксированный val holdout** из отдельного `val_seed` |
| CE ≈ const → бессмысленные корреляции | **Tier C**: дисперсия / диапазон `val_ce` по эпохам должны быть ненулевыми (есть динамика) |

**Ядро (`wfr_core.py`, `wfr_lm.py`, `wfr_rfp.py`) не меняется** — только новый тренировочный цикл в `protocol_train.py`.

## Tier-ы

- **A (инженерия):** все веса finite; `max_grad_l2` ниже мягкого потолка (по умолчанию 25).
- **B (поле):** `final_val_rc` не коллапсирует ниже порога (по умолчанию 0.35); spike не обязан быть в коридоре (toy).
- **C (сигнал задачи):** на фиксированном val временной ряд `val_ce` не плоский (есть `std` / `range` выше минимальных порогов).

## Локально

Из корня репозитория:

```powershell
python experiments/06-rfp-protocol-tests/run_tiered_suite.py --epochs 32 --rfp-version v03
python -m pytest experiments/06-rfp-protocol-tests/test_protocol.py -v
```

## Удалённый GPU (внутренний контур)

Параметры SSH — в `RULES.md` (секция 6): хост `10.100.9.10`, пользователь `alex1`, рабочая копия `~/Desktop/WFR-Memory-Test/` с venv `.venv/bin/python`.

### 1. Загрузить дерево (обновить пути при необходимости)

```powershell
$H = "ssh-ed25519 255 SHA256:qBpeuLXUWdtZt7kF1ZouvO+veIZ7nog2qToAcIUVbow"
$R = "c:\WFR-Architecture"

& "C:\Program Files\PuTTY\pscp.exe" -pw "1" -hostkey $H `
  "$R\wfr_rfp.py" "$R\wfr_lm.py" `
  "alex1@10.100.9.10:/home/alex1/Desktop/WFR-Memory-Test/"

& "C:\Program Files\PuTTY\pscp.exe" -pw "1" -hostkey $H `
  "$R\experiments\00-smoke-test\wfr_core.py" `
  "$R\experiments\00-smoke-test\phase0_best_config.py" `
  "alex1@10.100.9.10:/home/alex1/Desktop/WFR-Memory-Test/experiments/00-smoke-test/"

& "C:\Program Files\PuTTY\pscp.exe" -pw "1" -hostkey $H `
  "$R\experiments\06-rfp-v0\run_rfp_training.py" `
  "alex1@10.100.9.10:/home/alex1/Desktop/WFR-Memory-Test/experiments/06-rfp-v0/"

& "C:\Program Files\PuTTY\pscp.exe" -pw "1" -hostkey $H `
  "$R\experiments\06-rfp-protocol-tests\protocol_train.py" `
  "$R\experiments\06-rfp-protocol-tests\tier_checks.py" `
  "$R\experiments\06-rfp-protocol-tests\run_tiered_suite.py" `
  "$R\experiments\06-rfp-protocol-tests\test_protocol.py" `
  "alex1@10.100.9.10:/home/alex1/Desktop/WFR-Memory-Test/experiments/06-rfp-protocol-tests/"
```

### 2. Запуск suite на GPU

```powershell
& "C:\Program Files\PuTTY\plink.exe" -ssh -batch -pw "1" -hostkey "ssh-ed25519 255 SHA256:qBpeuLXUWdtZt7kF1ZouvO+veIZ7nog2qToAcIUVbow" alex1@10.100.9.10 "cd ~/Desktop/WFR-Memory-Test && mkdir -p experiments/06-rfp-protocol-tests/outputs && PYTHONPATH=. .venv/bin/python experiments/06-rfp-protocol-tests/run_tiered_suite.py --epochs 32 --rfp-version v03 2>&1 | tee experiments/06-rfp-protocol-tests/outputs/run_tiered_suite.log"
```

Долгий прогон — `nohup` по образцу из `RULES.md`.

### 3. Скачать результаты

```powershell
$H = "ssh-ed25519 255 SHA256:qBpeuLXUWdtZt7kF1ZouvO+veIZ7nog2qToAcIUVbow"
$OUT = "c:\WFR-Architecture\experiments\06-rfp-protocol-tests\outputs\remote_gpu"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
& "C:\Program Files\PuTTY\pscp.exe" -pw "1" -hostkey $H `
  "alex1@10.100.9.10:/home/alex1/Desktop/WFR-Memory-Test/experiments/06-rfp-protocol-tests/outputs/*" `
  "$OUT\"
```

## Артефакты

- `outputs/tiered_suite_report_*.json` — baseline vs RFP, tier A/B/C, `overall_pass`.
- PNG кривых (если не `--no-png`): `outputs/protocol06_*.png`.

## Связь с документацией

См. `docs/11-rfp-v0-spec.md` (общий контекст RFP). Этот каталог — **отдельный протокол оценки**, не замена Exp 06 grid.
