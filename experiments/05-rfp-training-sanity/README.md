# Experiment 05 — RFP Training Sanity (Phase 1, первый проверочный тест)

**Дата:** 31 марта 2026  
**Статус:** Активный (точка входа в Phase 1)

**Краткий итог прогона 2026-03-31:** enhanced, `--epochs 30`, GPU A100 — precheck OK, **PASS** (снижение val total/CE); полная таблица и ссылка на `docs/07-experiment-plan.md` (Test 3) — ниже в разделе про удалённый GPU.

## Цель

Проверить предпосылки обучения из [`docs/10-phase-1-plan.md`](../../docs/10-phase-1-plan.md) и раздел 6 [`docs/03-theory.md`](../../docs/03-theory.md):

- градиенты проходят через `WFRNetwork` (WPE, резонанс, surrogate spike);
- на toy **next-token** задаче loss улучшается на **фиксированных** train/val батчах;
- нет NaN/Inf;
- в режиме **enhanced** — согласованность с **полной скалярной целью** \(L = \alpha L_{\text{task}} + \beta(1-\overline{\text{RC}}) + \gamma L_{\text{energy}}\).

Полноценный RFP (два режима, \(\Delta f,\Delta\theta\) из теории) — **не** этот скрипт; это инженерный sanity перед ним.

## Связь с Phase 0 и математикой

- **Единый конфиг резонанса:** частоты, пороги и режим `target_mode` для слоёв берутся из [`experiments/00-smoke-test/phase0_best_config.py`](../00-smoke-test/phase0_best_config.py) (`PHASE0_FREQ_BALANCED`). В коде: `BEST_CONFIG = PHASE0_FREQ_BALANCED` в `run_training_sanity.py`.
- **Формула \(L\):** реализация вынесена в `wfr_losses.py` (`compute_loss`, `task_loss_ce`, `rc_penalty`, `energy_cost`), чтобы совпадать с §6 [`docs/03-theory.md`](../../docs/03-theory.md) без дублирования.
- **Precheck до обучения** (`phase1_checks.py`): если не передан `--skip-precheck`, перед стартом эпох выполняются проверки:
  1. **phase0_params_in_wfr** — у построенной сети частоты/пороги/`target_mode` совпадают с `phase0_best_config` (регрессия вроде ошибки знака/несоответствия конфигов между фазами).
  2. **homeostatic_sign** — изолированный слой в `eval`: при завышенном пороге и низкой активности порог после шага homeostatic **снижается** (отрицательная обратная связь \(\Delta\theta = \eta(r_{\text{real}} - r_{\text{target}})\)).
  3. **loss_formula** — численное согласование `compute_loss` с ручным \(\alpha\cdot\text{CE} + \beta\cdot(1-\text{RC}) + \gamma\cdot\text{energy}\).

При падении любой из этих проверок скрипт завершается с **кодом 2**, обучение не запускается.

## Режимы

### По умолчанию — `enhanced`

- **Целевая функция:** \(L = \alpha \cdot \text{CE} + \beta \cdot (1 - \text{RC}) + \gamma \cdot \bar{\rho}_{\text{spike}}\), где \(\bar{\rho}_{\text{spike}}\) — средняя доля спайков по резонансным слоям (через surrogate).
- **Данные:** фиксированные `NUM_TRAIN_BATCHES` train-батчей и `NUM_VAL_BATCHES` val-батчей (разные сиды), повторяются каждую эпоху.
- **Обучение:** по умолчанию **120 эпох** (`EPOCHS` в скрипте), полный проход по train-батчам за эпоху. Параметр `--epochs N` переопределяет число эпох.
- **Метрики:** `val` **каждую эпоху**; в JSON — полные ряды `history.*_per_epoch`, лучшие **best_val_ce** / **best_val_total** и номера эпох; норма градиента (L2) **до** `optimizer.step()` на первом батче.
- **PNG:** `training_sanity_enhanced_curves_<timestamp>.png` (train / val total / val CE, \(\ln V\)) и **`training_sanity_enhanced_val_components_<timestamp>.png`** — разложение val: CE, \((1-\overline{\text{RC}})\), energy (§6). Файлы **не** в `.gitignore`.

- **JSON / история:** помимо `val_total` / `val_ce`, в `history` сохраняются `val_rc_term_per_epoch`, `val_energy_per_epoch`, а при обучении по полной L — `train_rc_term_per_epoch`, `train_energy_per_epoch`.

Константы \(\alpha,\beta,\gamma\) и размеры — в начале `run_training_sanity.py`.

### Полноценный протокол Phase 1 — `run_full_training.py`

Соответствует **§6** [`docs/03-theory.md`](../../docs/03-theory.md) и п. **3.2 (желательное)** [`docs/10-phase-1-plan.md`](../../docs/10-phase-1-plan.md): явное сравнение **полной** \(L\) на train с обучением **только по CE** на тех же фиксированных батчах.

1. **Precheck** (один раз).
2. **Прогон A:** полная целевая функция на train (как enhanced).
3. **Прогон B:** градиент только от CE; на val по-прежнему считается полное разложение \(L\) (как ведёт себя поле при task-only).
4. **Артефакты:** `training_full_protocol_<timestamp>.json` (вложенные `run_A_full_L`, `run_B_ce_only_train`, блок `comparison`), `training_full_protocol_compare_<timestamp>.png` (val total L и val CE для обоих прогонов).

По умолчанию: **120 эпох**, для прогона A включён критерий **`--strict`** (лучший val CE \< \(\ln V - 0.02\)); отключить: `--no-strict`. Только прогон A без B: `--skip-ce-baseline`.

```bash
python experiments/05-rfp-training-sanity/run_full_training.py
python experiments/05-rfp-training-sanity/run_full_training.py --epochs 30 --no-strict
```

**Код выхода:** `0` — PASS прогона A; `1` — FAIL прогона A; `2` — precheck.

### `--task-only` (в `run_training_sanity.py`)

Один прогон enhanced, но шаг оптимизации только по CE; val-метрики с полным разложением (как прогон B в протоколе выше). Удобно для отладки без второго полного прогона.

### `--quick`

Один фиксированный батч, **только CE**, **280 шагов**; полная кривая `loss_curve` в JSON и PNG `training_sanity_quick_curve_<timestamp>.png`.

## Критерии PASS

### enhanced

1. Все средние loss по эпохам конечны.
2. После первого `backward` градиент ненулевой (max abs > 1e-12).
3. Улучшение на **val** по одному из условий: снижение **total** \(L\) к концу обучения, или снижение **CE** к концу, или **CE** на последней эпохе ниже \(\ln(V)-0.02\), или **лучший** val CE за эпоху ниже \(\ln(V)-0.02\).

С флагом **`--strict`** дополнительно требуется, чтобы **лучший** val CE был ниже \(\ln(V)-0.02\) (жёсткая проверка «лучше случайного угадывания»); без `--strict` этот пункт только логируется в JSON.

### quick

Как раньше: конечный loss лучше начального или ниже \(\ln(V)-0.05\), градиенты ок.

## Запуск

```bash
# Только precheck (без полного обучения)
python experiments/05-rfp-training-sanity/phase1_checks.py

# Полная L + train/val (по умолчанию 120 эпох)
python experiments/05-rfp-training-sanity/run_training_sanity.py

# Укороченный прогон (например отладка)
python experiments/05-rfp-training-sanity/run_training_sanity.py --epochs 30

# Жёсткий PASS: лучший val CE < ln(V)−0.02
python experiments/05-rfp-training-sanity/run_training_sanity.py --strict

# Пропустить precheck (отладка ядра / ускорение)
python experiments/05-rfp-training-sanity/run_training_sanity.py --skip-precheck

# Быстро, только CE
python experiments/05-rfp-training-sanity/run_training_sanity.py --quick

# Enhanced, но оптимизация только CE (val — полное L)
python experiments/05-rfp-training-sanity/run_training_sanity.py --task-only --epochs 30

# Полноценный протокол §6 + сравнение с CE-only (см. раздел выше)
python experiments/05-rfp-training-sanity/run_full_training.py
```

**Коды выхода:** `0` — успех по критериям режима; `1` — ошибка выполнения; `2` — провал precheck (математика/Phase 0).

Артефакты в `outputs/`: `training_sanity_enhanced_*.json` + `*_curves_*.png` + `*_val_components_*.png`; `training_full_protocol_*.json` + `training_full_protocol_compare_*.png`; `training_sanity_quick_*.json` + `*_quick_curve_*.png`.

### Прогон на удалённом GPU A100 80GB (по локальному `RULES.md`)

**Workflow (секция 6 RULES):** в плоскую папку на сервере положить **`wfr_core.py`**, **`phase0_best_config.py`**, **`run_training_sanity.py`**, **`run_full_training.py`**, **`wfr_losses.py`**, **`phase1_checks.py`**. Выполнять только **`/.venv/bin/python`**. Логи и JSON сохранять в `outputs/` на сервере и **скачать** в `experiments/05-rfp-training-sanity/outputs/`.

**Прогон 2026-03-31:** `NVIDIA A100 80GB PCIe`, **precheck** перед обучением — OK; enhanced **PASS** (`--epochs 30`).

| Режим | Val total \(L\) (нач → кон) | Val CE (лучший) | Train total (1-я → последняя эпоха) | Grad L2 (шаг 0) | Артефакты |
|--------|-----------------------------|-----------------|--------------------------------------|-----------------|-----------|
| enhanced | 6.264 → 3.571 | best 3.534 @ ep 14 | 4.858 → 3.594 | 4.034 | `training_sanity_enhanced_20260331_1630.json`, `training_sanity_enhanced_curves_20260331_1630.png`, `run_training_sanity_gpu_20260331.log` |
| enhanced (ранее) | 6.264 → 3.572 | 6.264 → 3.572 | 4.858 → 3.581 | 4.034 | `training_sanity_enhanced_20260331_1604.json`, `training_sanity_enhanced_gpu.log` |
| quick | CE 6.149 → 3.468 | — | — | да | `training_sanity_quick_20260331_1604.json`, `training_sanity_quick_gpu.log` |

На val в начале **energy** ≈ 0 (средняя доля surrogate-спайков на необученной сети); **rc_term** остаётся малым — основной вклад в \(L\) даёт CE. Критерий `val_ce_below_random` для enhanced на этом прогоне **false** (CE конечный ~3.57 всё ещё чуть выше \(\ln 32\)), PASS выполнен за счёт **снижения val total и val CE**.

## Связь с теорией

- **enhanced** проверяет **три члена** целевой функции из §6 (task, устойчивость через RC, «энергия» спайков) в одном скаляре \(L\).
- **quick** оставлен для скорости CI и проверки только градиента CE.
- **precheck** отделяет регрессии реализации (знак homeostatic, совпадение с Phase 0, формула \(L\)) от стохастики обучения.
- **`run_full_training.py`** фиксирует сравнение полной \(L\) и CE-only на одних данных — опора для RFP v0 и пункта 3.2 Phase 1.
