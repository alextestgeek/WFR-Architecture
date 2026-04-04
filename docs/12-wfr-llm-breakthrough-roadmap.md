# 12. Дорожная карта: от WFR-теории к прорыву в LLM

**Роль документа:** план **автора теории**, который сознательно **не** продаёт «готовую теорию до масштаба». Теория и ядро **растут поэтапно**: каждый этап даёт **фальсифицируемое** предсказание; отказ от предсказания **меняет теорию или архитектуру**, а не только гиперпараметры.

**Связь:** Phase 0–1 закрыты; Phase 2 (WikiText, RFP, twin CE-only) — активная калибровка; этот документ — **Phase 3+ к прорыву**.

---

## 0. Принципы (нерушимые)

1. **Один эталонный протокол данных** на корпусе: fresh train / фиксированный val (как Exp 08 + protocol_train), одинаковые окна по сидам — для любой архитектуры.
2. **Сначала parity, потом «магия WFR»**: без честного baseline прорыв не доказуем.
3. **Раздельные цели обучения в отчётах**: full \(L\) vs CE-only (уже в Exp 08 Phase 2); val RC трактовать с оговоркой.
4. **Теория достраивается после этапа**: новая лемма / правило RFP появляется **только** если закрывает зазор между экспериментом и объяснением (см. §6).

---

## Этап A — Parity layer (инженерная истина)

**Цель:** на **одном** корпусе и бюджете шагов понять, теряем ли мы качество **из‑за специфики WFR** или из‑за общей недостаточности ёмкости / обучения.

| Действие | Критерий выхода |
|----------|------------------|
| A1. Минимальный **Transformer (causal)** на char WikiText-2, тот же протокол окон, тот же val seed | Скрипт в репо, воспроизводимый `best_val_ce` |
| A2. Сопоставимый по порядку **число параметров / шагов** с WFRLM прогоном (документировать расхождение, если 1:1 невозможно) | Таблица: TF vs WFR, CE, wall-clock |
| A3. Фиксация зазора | Если WFR сильно хуже при сопоставимом бюджете — приоритет **ёмкости readout / представления**, не RFP |

**Статус (2026-04):** **A1–A2 закрыты** скриптами Exp 09. **A3 / B1:** на A100 при **`--fair-parity`** узкий readout (**3**) даёт большой зазор по val CE; при **16** и 1:1 params зазор ~**0.05**; sweep **3–32** — при **readout 32** WFRLM **лучше** TinyTransformer по best val CE (Δ отрицательный). Приоритет **ёмкости readout**, не RFP. Каноничные артефакты разнесены по прогонам: [`experiments/09-lm-parity/outputs/remote_a100/README.md`](../experiments/09-lm-parity/outputs/remote_a100/README.md); сводная таблица — [`experiments/09-lm-parity/README.md`](../experiments/09-lm-parity/README.md).

Инструменты: **`--match-capacity`**, **`--match-optimizer`**, пресет **`--fair-parity`**; **`--readout-feat-dim`**, **`--readout-mlp-hidden`**, **`--readout-wave-kernel`** для B1; **`--phase-causal-kernel`**, **`--content-neighbor-mix`** для B3; удалённый батч **`remote_gpu_parity.sh`** (sweep **`PARITY_READOUT_DIMS`**; опционально **`PARITY_READOUT_MLP_HIDDEN`**, **`PARITY_READOUT_WAVE_KERNEL`**, **`PARITY_PHASE_CAUSAL_KERNEL`**, **`PARITY_CONTENT_NEIGHBOR_MIX=1`**).

**B3 на A100 (2026-04-03):** [`remote_gpu_b3_confirm.sh`](../experiments/09-lm-parity/remote_gpu_b3_confirm.sh), `readout_feat_dim=16`, 48 ep, fair-parity. **`--phase-causal-kernel 3`:** Δ ≈ **0** (WFRLM на уровне matched TF). **`--readout-wave-kernel 3`:** зазор **меньше**, чем у контроля (+0.047 → +0.034 по CE). Локальный `--quick` с `readout=3` давал иной масштаб эффекта — не экстраполировать без повторной калибровки. Детали: [`docs/13-project-status-snapshot.md`](13-project-status-snapshot.md), [`experiments/09-lm-parity/outputs/remote_a100/runs/03_b3_confirm_20260403/`](../experiments/09-lm-parity/outputs/remote_a100/runs/03_b3_confirm_20260403/).

---

## Этап B — Ёмкость под язык (архитектура)

**Цель:** устранить узкие горлышки, **не** ломая доказуемые Phase 0 свойства там, где они реальны.

| Действие | Связь с теорией |
|----------|------------------|
| B1. Расширить readout и/или промежуточную размерность **наблюдаемо** | Гипотеза: информация о токене не умещается в линейный канал к логитам; в коде: `WFRLM(readout_feat_dim, readout_mlp_hidden)`, CLI `--readout-feat-dim`, `--readout-mlp-hidden`; Exp 09 + `remote_gpu_parity.sh` (`PARITY_READOUT_MLP_HIDDEN`) |
| B2. Subword / BPE токенизация | Та же физика позиций; меняется алфавит задачи |
| B3. Абляции: фазы, число слоёв, phase-lock; локальный контент и локальные фазы | Предсказание: отключение X ухудшит метрику Y на фиксированном корпусе; `content_neighbor_mix` — сосед в контенте; `phase_causal_kernel` — каузальная depthwise-смесь по времени на фазах в `WFRNetwork` (`run_wikitext_train --phase-causal-kernel`, §9.1 `03-theory.md`) |

**Выход:** зарегистрированный в `03-theory.md` или отдельном разделе **патч теории** только после 2+ согласующихся прогонов.

---

## Этап C — Масштаб и железо

**Цель:** отличить «игрушку» от траектории к LLM.

- Больше данных / длиннее train; при необходимости multi-GPU.
- Критерий: кривая val CE **продолжает** падать в режиме, где baseline TF тоже учится (скользящее сравнение).

---

## Этап D — Дифференциация по длине контекста

**Цель:** там, где заявление WFR сильнее всего (память на токен / стабильность фазы), получить **измеримое преимущество**: на одной задаче LM при \(T_1 \ll T_2\) сравнить качество и **реальный** cost (время, память) против baseline.

**Условие честности:** одинаковый budget на «полезные» шаги или нормировка по FLOPs-порядку (грубо допустимо).

### D.1 Черновик протокола (реализация следующим PR / скриптом)

1. **Корпус и сиды:** тот же WikiText-2 char, те же `val_seed`, что Exp 08–09; фиксировать fingerprint в JSON.
2. **Два режима длины:** короткое окно \(T_1\) (например 96, как сейчас в parity) и длинное \(T_2\) (например 512 или 1024 — выбрать так, чтобы TF с **полным** causal self-attention по всей длине оставался на GPU без OOM при выбранном `d_model`; иначе явно зафиксировать TF с **скользящим окном** attention `max_len = T_2` или сравнимым ограничением).
3. **Matched TinyTransformer:** `--match-capacity` к WFRLM при **одинаковом** \(T\) в паре (отдельный match на \(T_1\) и на \(T_2\) или один TF с `max_len ≥ T_2` — документировать).
4. **Метрики в отчёте:** `best_val_ce` (как в A2), **peak GPU memory** (torch.cuda.max_memory_allocated или nvml), **wall-clock на эпоху** (среднее по последним \(k\) эпохам).
5. **Предсказание до прогона:** WFR не хуже TF по CE при более выгодном соотношении память/время на \(T_2\) — либо формулировка опровержима (хуже по обоим — пересмотр заявки этапа D).

Кодовая база (черновик): [`run_longcontext_pair.py`](../experiments/09-lm-parity/run_longcontext_pair.py) — несколько `--seq-lens`, для каждого полный fair-parity прогон, в JSON **`d1_metrics`**: `wall_seconds_full_pair`, `peak_cuda_mib` (если CUDA). Пример:  
`python experiments/09-lm-parity/run_longcontext_pair.py --fair-parity --seq-lens 96,512 --epochs 48 --num-train-batches 20 --num-val-batches 8 --batch-size 16 --readout-feat-dim 32`  
Проверка структуры сохранённых JSON: [`verify_longcontext_artifacts.py`](../experiments/09-lm-parity/verify_longcontext_artifacts.py).

---

## Этап E — RFP как теория обучения поля

**Цель:** не «улучшить CE иногда», а **предсказуемое** правило: при условиях \(C\) шаг RFP **улучшает** метрику \(M\) с вероятностью \(p\) в измеримом коридоре.

**Если после B–C RFP не даёт устойчивого эффекта** — теория RFP для LLM **сужается** (например, только для определённой шкалы или отключается в пользу чистого autograd до новой гипотезы).

---

## Этап F — Внешняя проверка (прорыв по индустрии)

- Стандартные или близкие к стандартным **held-out** метрики на том же классе моделей.
- Публикация уровня: воспроизводимый код + кривые + честные ограничения.

---

## 6. Петля «эксперимент → ядро → теория»

1. Формулируем **предсказание** до прогона (одним абзацем в журнале эксперимента).
2. Прогон → JSON/лог.
3. **Pass:** теория непротиворечива; опционально добавить предсказательное следствие в `03-theory.md`.
4. **Fail:** либо баг/протокол, либо **ревизия** гипотезы (явно в docs, не только в чате).

---

## Текущий фокус (что делаем сейчас)

Сквозной порядок доказательств (**модель работает · контекст→волна · точность LM · обучаемость**): [`07-experiment-plan.md`](07-experiment-plan.md) — раздел **Phase 3+**.

Детальный чеклист, матрица гипотез **H1–H8** и узкие места **U1–U6:** [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md).

1. **Регрессия ядра:** после правок `wfr/core.py` — smoke Phase 0 (`experiments/00-smoke-test/run_smoke_test.py`).
2. **B1:** **готово** — [`runs/04_b1_mlp_matrix_20260403`](../experiments/09-lm-parity/outputs/remote_a100/runs/04_b1_mlp_matrix_20260403); см. **H2** в [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md).
3. **B3 (combo):** `readout_wave_kernel=3` + **MLP 64** при **D=32**, fair-parity 48 ep — скрипт [`remote_gpu_b3_combo_d32.sh`](../experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh); после прогона — JSON в `outputs/`, разнести в `runs/` при необходимости.
4. **C:** удлинить train при лучшей связке из п.3; сравнение кривых val CE с тем же TinyTransformer.
5. **D:** замеры по **D.1** — на A100: [`remote_gpu_longcontext_d32.sh`](../experiments/09-lm-parity/remote_gpu_longcontext_d32.sh) (пары длин, readout по умолчанию 32); проверка JSON: [`verify_longcontext_artifacts.py`](../experiments/09-lm-parity/verify_longcontext_artifacts.py).
6. **RFP (E):** не как главный рычаг для базового зазора CE — после стабилизации B/C.

`05-roadmap.md` уже ссылается на Exp 09 и этот документ.
