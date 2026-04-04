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

## Удалённый GPU

Пара **не** коммитится в git: хост, пользователь, пароль, SSH host key и пути на сервере храните **только локально** (как в [`experiments/06-rfp-v0/REMOTE_RUN.md`](../06-rfp-v0/REMOTE_RUN.md) — там шаблон с `<HOST>`, `<PASSWORD>`, `RULES.md`).

Краткий чеклист:

1. Скопировать на сервер: `wfr_lm.py`, `wfr_rfp.py`, `experiments/00-smoke-test/wfr_core.py`, `experiments/00-smoke-test/phase0_best_config.py`, `experiments/06-rfp-v0/run_rfp_training.py`, файлы из `experiments/06-rfp-protocol-tests/` (`protocol_train.py`, `tier_checks.py`, `run_tiered_suite.py`, `test_protocol.py`).
2. На сервере из корня рабочей копии (venv с PyTorch):  
   `PYTHONPATH=. .venv/bin/python experiments/06-rfp-protocol-tests/run_tiered_suite.py --epochs 32 --rfp-version v03`
3. Скачать `experiments/06-rfp-protocol-tests/outputs/*` в локальный каталог, например `experiments/06-rfp-protocol-tests/outputs/remote_gpu`.

Для синхронизации целого дерева с переменными окружения см. `experiments/08-wikitext-rfp/remote_sync_wikitext.ps1` (там те же `WFR_SSH_*`).

## Артефакты

- `outputs/tiered_suite_report_*.json` — baseline vs RFP, tier A/B/C, `overall_pass`.
- PNG кривых (если не `--no-png`): `outputs/protocol06_*.png`.

## Связь с документацией

См. `docs/11-rfp-v0-spec.md` (общий контекст RFP). Этот каталог — **отдельный протокол оценки**, не замена Exp 06 grid.
