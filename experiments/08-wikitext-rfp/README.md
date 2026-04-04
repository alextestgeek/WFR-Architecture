# Experiment 08 — WikiText-2 (raw), char-level LM + RFP

## Гипотеза

На **реальном** тексте (не синтетический toy) тот же стек, что в Exp 05–07 (Adam + композитный лосс + RFP), **обучается по шагам** без коллапса: val/train CE, RC и spike rate отслеживаются по эпохам; сравнение **baseline vs RFP** с PNG и JSON.

Это **не** claim про SOTA по перплексии — цель: калибровка на малом корпусе и воспроизводимый протокол для Phase 2+.

## Данные

- Hugging Face `wikitext / wikitext-2-raw-v1`, локально: `data/hf/wikitext-2-raw-v1/` (`datasets.save_to_disk`).
- Скачивание из корня репозитория:

```powershell
pip install datasets
python data/hf/download_wikitext2.py
```

## Файлы

| Файл | Назначение |
|------|------------|
| `wikitext_loader.py` | Char-vocab с train, окна для train/val |
| `run_wikitext_train.py` | Основной цикл: fresh train each epoch, фиксированный val |
| `test_wikitext_smoke.py` | Pytest: корпус + 1 эпоха baseline + PNG/JSON в `outputs/` |
| `test_wikitext_integrity.py` | Pytest: val holdout воспроизводим, train окна меняются между эпохами |
| `wikitext_integrity.py` | SHA256 отпечаток батчей + assert’ы (без подмены val) |
| `test_wikitext_ab.py` | A/B Adam vs RFP v02/v03 + столбчатая диаграмма |
| `remote_sync_wikitext.ps1` | Загрузка артефактов на удалённый GPU (см. ниже) |
| `test_input_schemes.py` | Сравнение входных схем (`absolute`, `token_as_pos`, `content_off`, …), много сидов; PNG динамики по эпохам |
| `run_theory_phase2.py` | Один suite «теория ↔ тест»: A/B RFP, head-to-head позиций, контроль `content_off`, CE+RC/spike, bar PNG |

## Запуск локально

```powershell
cd c:\WFR-Architecture
.venv\Scripts\python experiments\08-wikitext-rfp\run_wikitext_train.py --quick
.venv\Scripts\python experiments\08-wikitext-rfp\test_wikitext_ab.py --quick
pytest experiments\08-wikitext-rfp\test_wikitext_smoke.py -v
```

Артефакты по умолчанию: `outputs/wikitext08_*.png`, `wikitext08_last_run.json`, `wikitext08_epoch_log.json`, `wikitext08_vocab_meta.json`.

### Длинный прогон (явные эпохи, CSV, манифест)

```powershell
python experiments\08-wikitext-rfp\run_wikitext_train.py --long-run
```

- **72 эпохи**, seq **96**, **20** train-батчей на эпоху, **8** val-батчей; печать **таблицы** в консоль; `wikitext08_epochs_<timestamp>.csv`; `wikitext08_manifest_<timestamp>.json` с формулой лосса (`ALPHA`/`BETA`/`GAMMA` из кода), сидами, **SHA256 val holdout**, итоговыми метриками.

### Дожим WikiText на большой GPU (`--wiki-push`)

Один флаг задаёт тяжёлый прогон: **200 эпох**, seq **256**, batch **64**, **64** train- и **24** val-батча на эпоху, **8** резонансных слоёв (частоты/порога — Phase0 для первых четырёх, дальше геометрическое продолжение). Артефакты с меткой `wikipush_*_<timestamp>`.

```powershell
python experiments\08-wikitext-rfp\run_wikitext_train.py --wiki-push --rfp-version v03
```

Другое число слоёв: `--num-resonance-layers 12`. При OOM уменьшите batch вручную (пока без отдельного флага — правьте константы `WIKI_PUSH_*` в `run_wikitext_train.py` или запускайте без `--wiki-push`, передав `--epochs`, `--batch-size`, `--seq-len`, `--num-train-batches` сами).
- Отключить проверки целостности (только отладка): `--no-integrity`.
- Тише консоль: `--long-run --quiet-epochs`.

### Схемы ввода (pos/content) + динамика на A100

`test_input_schemes.py` пишет для **каждого** прогона `*_curves.png` (как в основном тренере: CE/RC/spike) и отдельно **`*_dynamics.png`**: val CE + best-so-far, **val acc@1**, train CE, val RC и spike по эпохам. JSON отчёт содержит `per_run` с путями к CSV/PNG.

```powershell
# Короткий скрининг
python experiments\08-wikitext-rfp\test_input_schemes.py --quick

# Длинный GPU-прогон: --long поднимает нижнюю границу минут и объём (эпохи, батчи, seq); для всех 5 схем добавьте --schemes full
python experiments\08-wikitext-rfp\test_input_schemes.py --long --minutes 45 --schemes full --seeds 42,43
```

Пороги по `--minutes`: от ~25 мин — до 72+ эпох и seq 112+; от 40+ — 96 эпох, seq 128, больше train/val батчей; от 55+ — до 120 эпох.

### Phase 2 — теория и развязка факторов (`run_theory_phase2.py`)

Один скрипт закрывает приоритетный протокол: **сначала** Adam vs RFP на `absolute/normal`, **затем** сравнение `absolute` vs `token_as_pos` при RFP (ветка `absolute/normal+RFP` не дублируется — метрики берутся из первой фазы), **в конце** контроль **`absolute/off`**.

**Гипотезы (фиксируются в JSON):**

- **H1:** RFP улучшает или стабилизирует val CE относительно Adam-only при той же входной схеме.
- **H2:** Корректная разметка позиций (`absolute`) сопоставима с `token_as_pos` при достаточной длине обучения.
- **H3:** Отключение content (`off`) даёт контролируемую деградацию; улучшение CE проверяется вместе с **RC/spike** (поля `wave_health_*` в `per_run`).

**Критерии успеха:** нет NaN; `best_val_ce` заметно ниже ln(V); по нескольким сидам видны воспроизводимые различия; на эпохе минимума val CE spike/RC не выглядят тривиально «мёртвыми» (`wave_health_at_best`).

```powershell
python experiments\08-wikitext-rfp\run_theory_phase2.py --quick
python experiments\08-wikitext-rfp\run_theory_phase2.py --minutes 90 --seeds 42,43,44
```

Артефакты: `outputs/wikitext08_phase2_<timestamp>.json`, `wikitext08_phase2_bar_{ab_rfp,h2h_pos,control_off}_<timestamp>.png`, для каждого прогона `phase2_<phase>_*_dynamics.png` и CSV.

### Целостность (нет «подкрутки» val)

1. **Val:** один раз строится из `val_seed`; отпечаток тензоров в манифесте; assert — повторная сборка даёт тот же SHA256.
2. **Train:** окна эпохи 0 и 1 **обязаны различаться** (fresh train).
3. **Инициализация:** `--init-seed` (по умолчанию 42) фиксирует стартовые веса; метрики считаются из CE/RC/energy по определениям в коде, без скрытых бонусов к val.

## Протокол (по шагам)

1. **Эпоха N:** новые случайные окна из **train** (`train_seed + N * 100_003`).
2. **Val:** один и тот же holdout из **validation** (`val_seed`).
3. После прохода по train — **mean train CE** на тех же батчах (eval), затем **val CE / RC / spike**.
4. PNG: val CE + **train CE** (если доступен), RC, spike; линия **ln(V)** для ориентира.

## Удалённый GPU (закрытый контур)

Полный цикл (хост, пароль, `pscp`/`plink`, venv) — в **`RULES.md` §6–8** (файл локальный, в git не коммитится).

1. Задать `WFR_SSH_PASSWORD`, `WFR_SSH_HOSTKEY` и при необходимости хост/пользователя (см. заголовок `remote_sync_wikitext.ps1`).
2. Выполнить `remote_sync_wikitext.ps1 -Direction upload` — на сервер уходит **мини-дерево как в git** (`wfr/`, `experiments/00-smoke-test`, `06-rfp-v0`, `08-wikitext-rfp`, корневые `wfr_lm.py` / `wfr_rfp.py`). Плоская одна папка не подходит: нужен пакет `wfr` и корень репозитория на `sys.path`.
3. На сервере: venv с PyTorch CUDA + `pip install datasets matplotlib`; из корня синка: `python data/hf/download_wikitext2.py`, затем  
   `cd ~/Desktop/WFR-Memory-Test && .venv/bin/python experiments/08-wikitext-rfp/run_wikitext_train.py --quick` (или `run_theory_phase2.py`; путь к venv и каталогу поправьте под свою раскладку).
4. Скачать артефакты: `remote_sync_wikitext.ps1 -Direction download` или `pscp` с `.../experiments/08-wikitext-rfp/outputs/*`.

Секреты в репозиторий не кладём; пуш этого этапа по желанию политики проекта.

## Критерий успеха (инженерный)

- Скрипт завершается без NaN; `wikitext08_last_run.json` содержит разумные `best_val_ce`, `final_val_rc`.
- PNG кривые и (для A/B) bar-chart сохранены.

## Вердикт

- **Input schemes (full, remote 2026-04-02):** на 30 эп. и двух сидах `absolute/normal`, `absolute_modV` и `x0.5` дают сопоставимый best val CE (~2.99); `token_as_pos` чуть слабее; `absolute/off` хуже (~3.19). Детали: `outputs/wikitext08_input_schemes_20260402_111034.json`.
- **Phase 2:** см. `outputs/wikitext08_phase2_*.json` после `run_theory_phase2.py` на GPU. В отчёте: `sections.ab_rfp_full_L` vs `ab_rfp_ce_only_train`, плюс `paired_ab_rfp_objectives` (сравнение val CE и RC на лучшей эпохе для полного \(L\) и для train CE-only). Отключить дублирование: `--no-ce-only-twin`. Одиночный прогон: `run_wikitext_train.py --loss-mode ce_only`.
