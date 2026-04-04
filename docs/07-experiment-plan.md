# 07. Experiment Plan — План проверки теории

**Версия:** 0.6  
**Дата:** 3 апреля 2026  
**Статус:** Утверждён

**Сводный план Phase 0 (цели, соответствие теории, критерии выхода):** [`08-phase-0-plan.md`](08-phase-0-plan.md)

Мы начинаем проверку теории WFR **с логического начала** — с проверки жизнеспособности архитектуры, прежде чем углубляться в сложные механизмы обучения.

---

## Философия проверки

- Начинать с **простого**
- Сначала убедиться, что система **вообще работает** и даёт осмысленный выход
- Если направление подтверждается — постепенно усложнять
- В дальнейшем стремиться к эмуляции фотонного, нейроморфного и квантового оборудования (пока на GPU)
- Каждый значимый эксперимент фиксировать с выводами и версионностью

---

## Phase 0: Viability & Stability (Первые эксперименты)

### Test 0 — Smoke Test (Жизнеспособность)

**Цель:** Проверить, что базовая архитектура вообще запускается и формирует ожидаемые структуры.

**Статус:** Завершён (версия 2.1 — Theoretical Compliance)

**Результаты:**
- Базовая архитектура жизнеспособна
- Все 4 слоя могут быть активными одновременно при правильном подборе параметров
- Event-driven поведение подтверждено (много "молчаливых" нейронов)
- Лучшие конфигурации: `target_mode="frequency"` и `target_mode="mean"`

**Ограничения выявленные (v2.0, устранены в v2.1):**
- ~~Масштабируемость слабая (при >6–8 слоёв большинство уровней "мёртвые")~~ — причина: баг инвертированного знака в Homeostatic regulation. Исправлено: 32/32 слоёв активны.
- Параметры требуют тщательного подбора (Homeostatic regulation решает это автоматически)

**Документация:** [`docs/strategy-fractal-layers.md`](docs/strategy-fractal-layers.md)

### Memory & Complexity Test (Новый тест)

**Цель:** Экспериментально проверить декларации теории о сложности и памяти **до** перехода к обучению.

- Измерять память на токен и время forward pass
- Контексты: 512 → 4096 → 16384 → 65536 → 131072 → 262144
- Построить графики: Memory per token, Time vs Context Length, Log-log plot
- Использовать лучшие конфигурации из Test 0 (Freq-Balanced / Mean-Balanced)

**Критерии успеха:**
- Память на токен остаётся O(1) или O(log n)
- Время масштабируется лучше O(n)
- Результаты честно документируются (даже если они опровергают теорию)

**Файл:** `experiments/01-memory-complexity-test/run_memory_test.py`

**Статус:** **Завершён** — O(1) память подтверждена до 100M токенов. Результаты: [`09-memory-complexity-test-plan.md`](09-memory-complexity-test-plan.md)

### Layer Scaling Test (Experiment 02)

**Цель:** Проверить масштабирование до 32 фрактальных слоёв с механизмами стабилизации v2.1.

**Статус:** **Завершён** — 32/32 слоёв активны. Критический баг homeostatic найден и исправлен.

**Файл:** `experiments/02-layer-scaling-test/run_layer_scaling_test.py`

**Результаты:** Все 4 стратегии частот × 5 глубин (4–32) × 3 контекста (512, 8K, 131K). Подробности: [`03-theory.md`](03-theory.md) раздел 8.

### Test 1 — Long Context Stability (Experiment 03)

**Цель:** Проверить стабильность **внутренней структуры** паттернов при росте длины контекста.

**Статус:** **Завершён** — 5/6 суб-тестов пройдены (после исправления методологии). Единственный FAIL (spike distribution) — свойство необученных весов.

**Файл:** `experiments/03-long-context-stability/run_stability_test.py`

**Отличие от Memory Test:** Memory Test измерял агрегированные метрики (общий RC, общее время, память). Этот тест измеряет внутреннюю структуру:

- Тестировать длины: 512 → 4096 → 16384 → 65536 токенов
- 6 суб-тестов:
  1. **Determinism** — один вход → идентичный выход (max Δ < 1e-6)
  2. **Phase Encoding Stability** — фазы позиции p одинаковы при любом контексте
  3. **Cross-Context Standing Wave** — стоячая волна для общего префикса (cos sim > 0.95)
  4. **Windowed RC** — sliding window RC, нет ли мёртвых зон (CV < 0.15)
  5. **Spike Distribution** — равномерность спайков (max/min ratio < 3.0)
  6. **Depth Coherence** — RC не деградирует к концу (Q4/Q1 > 0.85)

**Критерии успеха:**
- Все 6 суб-тестов пройдены
- Паттерны не разрушаются при увеличении контекста
- Система остаётся стабильной

### Test 2 — Basic Pattern Formation (Experiment 04)

**Цель:** Проверить, что разные синтетические **порядки позиций** дают различимые стоячие волны / сигнатуры ([`03-theory.md`](03-theory.md), раздел 5), без обучения.

**Статус:** **Завершён** — различимость и повторяемость по порогам PASS; «STRONG» (<0.995) не достигнут.

**Файл:** `experiments/04-basic-pattern-formation/run_pattern_test.py`

**Метод:** шесть классов паттернов одной длины (linear, reverse, mod, stride, shuffle, two_blocks); сигнатура = профиль стоячей волны по четвертям + спайки + RC; warmup linear + заморозка homeostatic. Опционально: шум по фазам после WPE.

**Критерии (см. README эксперимента):** **PASS** по различимости — максимальный попарный косинус сигнатур разных классов < 0.99999 (нет почти полного совпадения нормированных векторов). **STRONG** — тот же максимум < 0.995; на фиксированном прогоне не выполнен (близкие пары вроде stride/shuffle). PASS не утверждает сильную попарную различимость во всех парах.

**Результаты:** [`experiments/04-basic-pattern-formation/README.md`](../experiments/04-basic-pattern-formation/README.md)

---

## Phase 1 — Обучение и RFP

**Мастер-план:** [`10-phase-1-plan.md`](10-phase-1-plan.md) (цели, критерии успеха, оговорки, реестр экспериментов).

### Test 3 — RFP Training Sanity (Experiment 05)

**Цель:** Проверить, что градиенты проходят через WFR (WPE, резонанс, surrogate spike) и что на toy next-token задаче loss снижается при фиксированных train/val батчах; до обучения — precheck согласованности с Phase 0, знака homeostatic и формулы \(L\).

**Статус:** Пройден — **короткий прогон enhanced** (`--epochs 30`, **2026-03-31**, NVIDIA A100 80GB, удалённый сервер; workflow в локальном `RULES.md`).

**Результаты (короткий тест):**

| Проверка | Итог |
|----------|------|
| Precheck (`phase0_params_in_wfr`, `homeostatic_sign`, `loss_formula`) | все OK |
| Устройство | `cuda` |
| Val total \(L\) | 6.264 → 3.571 (лучший **3.534** на эпохе 14) |
| Val CE | 6.264 → 3.571 (лучший **3.534** @ ep 14; \(\ln 32 \approx 3.466\)) |
| Градиент L2 (первый шаг) | ≈ 4.03 |
| PASS по критериям README | **да** (снижение val total и val CE) |
| Критерий `--strict` (лучший val CE < \(\ln V - 0.02\)) | **нет** (лучший CE чуть выше порога) |

**Артефакты:** [`training_sanity_enhanced_20260331_1630.json`](../experiments/05-rfp-training-sanity/outputs/training_sanity_enhanced_20260331_1630.json), `training_sanity_enhanced_curves_20260331_1630.png`, `run_training_sanity_gpu_20260331.log` — см. [`experiments/05-rfp-training-sanity/README.md`](../experiments/05-rfp-training-sanity/README.md).

**Критерии PASS:** см. README эксперимента (конечные метрики, ненулевые градиенты, снижение val loss; опционально `--strict`).

### Test 3b — Полноценный протокол Phase 1 (тот же Experiment 05)

**Цель:** зафиксировать обучение по полной целевой \(L\) из §6 и сравнить с обучением только по CE на идентичных данных ([`run_full_training.py`](../experiments/05-rfp-training-sanity/run_full_training.py), README эксперимента).

**Статус:** реализовано в репозитории; зафиксированный GPU-прогон — по мере выполнения (артефакты `training_full_protocol_*.json` в `outputs/`).

**Запуск:** `python experiments/05-rfp-training-sanity/run_full_training.py` (см. README эксперимента; для отладки: `--epochs 30 --no-strict`).

### Test 4 — RFP v0 (Experiment 06)

**Цель:** ввести **Resonant Field Plasticity v0** поверх Adam и целевой \(L\) из §6: явные \(\Delta\) для `frequency`, `phase_bias`, `decay` на `WFRNetwork`; контентный канал и readout в `WFRLM`; опционально `rfp_step_v01` (cos между слоями).

**Статус:** реализовано в репозитории; короткий A/B на фиксированных батчах — [`experiments/06-rfp-v0/outputs/ab_rfp_baseline.json`](../experiments/06-rfp-v0/outputs/ab_rfp_baseline.json). Полные primary-метрики (CE vs baseline, spike rate 0.18–0.32) — Phase 2, длинные прогоны.

**Документация:** [`11-rfp-v0-spec.md`](11-rfp-v0-spec.md), [`experiments/06-rfp-v0/README.md`](../experiments/06-rfp-v0/README.md).

**Запуск:** `python experiments/06-rfp-v0/test_rfp_vs_baseline.py --quick` (или без `--quick` для более длинного прогона).

### Experiment 08 — WikiText (char LM) + Phase 2 protocol

**Цель:** реальный корпус (WikiText-2 raw), сравнение входных схем и единый suite «теория ↔ измерения»: Adam vs RFP, `absolute` vs `token_as_pos`, контроль `content/off`, отчёт по CE и RC/spike.

**Статус:** протокол и скрипты в репозитории (`run_theory_phase2.py`, `test_input_schemes.py`); полные GPU-артефакты — в `experiments/08-wikitext-rfp/outputs/`.

**Документация:** [`experiments/08-wikitext-rfp/README.md`](../experiments/08-wikitext-rfp/README.md).

### Experiment 09 — LM parity (Transformer baseline, этап A дорожной карты)

**Цель:** тот же char WikiText и протокол окон, что Exp 08, плюс **минимальный causal Transformer** для честного зазора по val CE (`docs/12-wfr-llm-breakthrough-roadmap.md`, раздел A).

**Статус:** **A1–A3**; sweep readout **3/8/16/32**; **B1** (D=16/32 × linear + MLP64) — [`runs/04_b1_mlp_matrix_20260403`](../experiments/09-lm-parity/outputs/remote_a100/runs/04_b1_mlp_matrix_20260403) (**H2** в [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md)). **B3:** [`remote_gpu_b3_confirm.sh`](../experiments/09-lm-parity/remote_gpu_b3_confirm.sh) при `readout=16`. Реестр: [`outputs/remote_a100/README.md`](../experiments/09-lm-parity/outputs/remote_a100/README.md); срез: [`13-project-status-snapshot.md`](13-project-status-snapshot.md).

**Запуск:** `run_transformer_char_baseline.py` (A1); **`run_parity_pair.py --fair-parity`** (A2); sweep: `PARITY_READOUT_DIMS="3 8 16 32" bash remote_gpu_parity.sh`; B1: `bash remote_gpu_b1_mlp_matrix.sh`; B3: `bash remote_gpu_b3_confirm.sh`.

---

## Phase 3+ — План следующих шагов: **работа · волна · точность · обучаемость**

Задача: **доказуемо** (по протоколу и артефактам) связать четыре утверждения, которые вы отделяете от «маркетинга архитектуры».

| Опора | Что именно доказываем | Критерий «достаточно для нас» | Где снять | Порядок |
|--------|------------------------|-------------------------------|-----------|---------|
| **1. Модель работает** | Forward стабилен, выходы конечны, ядро + WFRLM собираются на целевом железе; регрессия не ломает Phase 0. | Smoke OK; при правках `wfr/core.py` — повтор `00-smoke-test` (+ по времени один тест из 01–02); для LM: один `--quick` parity или train без OOM. | `experiments/00-smoke-test/run_smoke_test.py`; `python -m pytest` по тестам репо; `run_parity_pair.py --quick --fair-parity` | **Постоянно** перед длинными GPU-прогонами |
| **2. Контекст формирует волну** | Длина/содержание входа **изменяет** наблюдаемое поле (фазы, стоячая волна, RC) предсказуемо; с токенами — отличие режима «есть контент» vs «только позиции». | Phase 0: стабильность префикса между длинами контекста (Exp 03), различимость синтетических порядков (Exp 04). LM-слой: `content_delta` off даёт иной профиль волны/CE чем on; при одном префиксе волна в зоне префикса близка (диагностика по желанию в JSON). | `experiments/03-long-context-stability/`; `experiments/04-basic-pattern-formation/`; Exp 08 абляция `content`; Exp 09/08 логи слоёв при необходимости | **Уже частично закрыто** (0–4); для LM — **короткий чеклист** после каждого крупного изменения readout/ядра |
| **3. Точность (LM)** | На **одном** корпусе и val CE модель не «ломается» относительно честного baseline при сопоставимом бюджете параметров. | Таблица fair-parity: Δ(WFR−TF); **B1** закрыт — [`runs/04_b1_mlp_matrix_20260403`](../experiments/09-lm-parity/outputs/remote_a100/runs/04_b1_mlp_matrix_20260403). Далее: этап **D** (**D.1** в [`12-wfr-llm-breakthrough-roadmap.md`](12-wfr-llm-breakthrough-roadmap.md)). | Exp 09, `remote_gpu_*` | **Сейчас:** combo B3 при D=32; реализация замеров D |
| **4. Обучаемость** | Градиенты проходят; CE (или согласованная цель) **падает** на фиксированных батчах/корпусе, без обвала RC «в ноль» как единственного объяснения. | Exp 05: ненулевой градиент, снижение val; Exp 08: кривые эпох; parity: лучший val CE лучше случайного уровня; при необходимости `--match-lr` / длиннее эпохи до вывода. | `experiments/05-rfp-training-sanity/`; `run_wikitext_train.py`; Exp 09 | **Параллельно** с B1; не путать с «прорывом RFP» |

**Сквозной порядок работ (практический):**

1. Регрессия **(опора 1)** после любых правок ядра/LM.
2. Быстрая проверка **(опора 2)** на WikiText: контент вкл/выкл — ожидаемо разный CE; при споре — отсылка к Exp 03–04.
3. Длинный GPU-прогон **(опора 3)** только с закрытым readout-контуром (B1/B3 по [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md)).
4. Журнал **(опора 4)** для каждого прогона: лучший val CE, число эпох до плато, при необходимости градиент на первом шаге (как в 05).

**Связь с теорией:** интерпретация «волна несёт контекст» для LM формализована в [`03-theory.md`](03-theory.md) §10–11; измеримые критерии не заменяют физическую картинку, а **фиксируют**, что картинка не противоречит метрикам.

### Чеклист из 10 шагов перед PR (`wfr/core.py`, `wfr_lm.py`, CLI Exp 08–09)

Все команды из **корня репозитория**. Минимум для «мелкого» PR — шаги **1–5**; при смене резонанса/фаз/стоячей волны добавить **7** (и при необходимости **8**).

Один прогон подряд (Windows): [`experiments/09-lm-parity/run_local_pr_checklist.ps1`](../experiments/09-lm-parity/run_local_pr_checklist.ps1) — шаги 1–6, опционально 7–8 (`-SkipOptional` отключает 7–8), затем `verify_theory_calibration.py` и `verify_longcontext_artifacts.py`.

| # | Команда | Порог PASS |
|---|---------|------------|
| 1 | `python -m compileall -q wfr wfr_lm.py wfr_rfp.py` | Код выхода 0, нет SyntaxError |
| 2 | `python experiments/00-smoke-test/run_smoke_test.py` | Завершение без исключения; в конце сообщение об успешном smoke |
| 3 | `python -m pytest -q --tb=short` | 0 failed (предупреждения вроде pytest-asyncio допустимы) |
| 4 | `python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity` | Код 0; в `experiments/09-lm-parity/outputs/parity_pair_*.json` поля `wfr_best_val_ce`, `transformer_best_val_ce` **конечны**, не NaN; датасет WikiText-2 в `data/hf/` по [`data/hf/README.md`](../data/hf/README.md) |
| 5 | `python experiments/09-lm-parity/run_transformer_char_baseline.py --quick` | Код 0; baseline JSON в `experiments/09-lm-parity/outputs/` |
| 6 | *(Опция, если менялся только readout/голова)* `python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity --readout-feat-dim 16` | Код 0; убедиться, что широкий readout не ломает пайплайн |
| 7 | *(Опция, если менялся `WFRNetwork` / standing wave / интерференция)* `python -m pytest experiments/03-long-context-stability/run_stability_test.py::test_1_determinism experiments/03-long-context-stability/run_stability_test.py::test_3_cross_context_standing_wave -q --tb=short` | Оба теста passed (test_3 — про префикс и волну при разной длине контекста) |
| 8 | *(Опция)* `python -m pytest experiments/08-wikitext-rfp/test_wikitext_smoke.py -q --tb=short` | passed (нужен корпус) |
| 9 | *(Только при спорных численных изменениях на CUDA)* загрузка по [`RULES.md`](../RULES.md) §6 + `experiments/08-wikitext-rfp/remote_sync_wikitext.ps1 -Direction upload`; на сервере: `bash ~/Desktop/WFR-Memory-Test/_remote_gpu_check.sh` (CUDA + `run_parity_pair.py --quick --fair-parity`) | В выводе: `CUDA True`, имя GPU; JSON `parity_pair_*.json` в `experiments/09-lm-parity/outputs/` на сервере; при необходимости `remote_sync_wikitext.ps1 -Direction download` |
| 10 | Если меняется **контракт** (новый флаг CLI, новая вводная в ядро): один абзац в PR + при необходимости патч [`03-theory.md`](03-theory.md) / README эксперимента | Ревьюер видит связь код ↔ док |

**Заметка:** шаг 4 уже покрывает **обучаемость** и **точность** в микро-бюджете; он **не** заменяет длинный A100-прогон для итоговой таблицы Δ.

---

## Дальнейшие шаги (после Phase 0)

- Phase 1: закрыт по плану — см. [`10-phase-1-plan.md`](10-phase-1-plan.md); Experiment 05 (sanity), Experiment 06 (RFP v0)
- Phase 2: Сравнение режимов WFR (WikiText, RFP, twin CE-only) — Exp 08
- **Phase 3 (прорыв LLM):** [`12-wfr-llm-breakthrough-roadmap.md`](12-wfr-llm-breakthrough-roadmap.md) — parity (Exp 09), ёмкость, масштаб, long-context diff, внешняя проверка. Операционная матрица узких мест и гипотез: [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md). Четыре опоры «работа / волна / точность / обучаемость»: **этот документ, раздел Phase 3+**.
- Phase 3 (оборудование): эмуляция специализированного оборудования — см. также [`05-roadmap.md`](05-roadmap.md)

---

**Следующий документ:** [05-roadmap.md](05-roadmap.md)

**Примечание:**  
Все результаты экспериментов будут фиксироваться в этом документе или в отдельных отчётах с указанием версии.
