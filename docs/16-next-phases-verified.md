# 16. Проверка тезисов (U1–U6, Phase 0, LM parity) и план следующих этапов

**Дата:** 31 марта 2026  
**Связь:** [`12-wfr-llm-breakthrough-roadmap.md`](12-wfr-llm-breakthrough-roadmap.md), [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md), [`experiments/09-lm-parity/README.md`](../experiments/09-lm-parity/README.md).

Этот документ **сверяет** с репозиторием формулировки про узкие места U1–U6 и «что уже доказано», затем даёт **один согласованный порядок работ** (без дублирования таблиц A100 — они остаются в Exp 09 и §14).

---

## 1. Верификация узких мест (матрица U1–U6)

| ID | Тезис | Обоснован? | Где зафиксировано / оговорка |
|----|--------|------------|-------------------------------|
| **U1** | Readout (поле → logits) — главный измеримый зазор при малом `readout_feat_dim`; рост D систематически сужает зазор с matched TinyTF | **Да** | Exp 09 sweep A100: D=3…32, таблица в [`09-lm-parity/README`](../experiments/09-lm-parity/README.md); **H1** подтверждена в [`14` §3](14-core-readiness-and-breakthrough-matrix.md). |
| **U2** | Один embedding токен→фазы может ограничивать «богатый» контент (BPE, смесь до стека) | **Как рабочая гипотеза**, не как численный факт | В [`14` §2 U2](14-core-readiness-and-breakthrough-matrix.md); **B2 / H8** в очереди — char vs BPE не сведены к одному отчёту. |
| **U3** | `phase_causal_kernel`, `content_neighbor_mix`, `readout_wave_kernel` при нормальном readout дают **тонкие** эффекты относительно расширения головы | **Согласуется с A100 B3** | Таблица подтверждения в Exp 09 README; **H3** ≈ ничья; **H4** слабое улучшение; локальный CPU quick может **не** совпадать по знаку Δ — в README явно предупреждено. |
| **U4** | Цель обучения: full L vs CE — нельзя смешивать выводы | **Да** | Принципы в [`12` §0 п.3](12-wfr-llm-breakthrough-roadmap.md); parity Exp 09 — `ce_only` для сравнения с TF. |
| **U5** | Бюджет эпох/данных — отдельный рычаг; малый бюджет не опровергает архитектуру | **Да** | **H5** / этап **C** в очереди [`14` §4](14-core-readiness-and-breakthrough-matrix.md). |
| **U6** | Дифференциация на длинном контексте требует протокола **D** (пары T, память TF, время) | **Частично реализовано в коде** | [`run_longcontext_pair.py`](../experiments/09-lm-parity/run_longcontext_pair.py) пишет JSON с `d1_metrics` (wall, peak CUDA); **полный отчёт D** (окно у TF, идентичный бюджет шагов) — ещё не сведён к каноничному «итоговому» прогону в `runs/`. |

**Вывод:** формулировки **U1, U3, U4, U5** хорошо опираются на уже имеющиеся артефакты; **U2** и **полная D-картина (U6)** остаются **очередью с явной постановкой**, а не «доказанностью».

---

## 2. Что уже хорошо измерено (согласно вашим протоколам)

| Блок | Статус проверки в репо |
|------|-------------------------|
| Phase 0 + память + глубина | Smoke, 01-memory, 02-layer, 03-stability, 04-pattern — пути в [`14` §1](14-core-readiness-and-breakthrough-matrix.md). |
| O(1) памяти на токен (постановка Memory Test) | [`09-memory-complexity-test-plan.md`](09-memory-complexity-test-plan.md); время ~O(n) не отменяется. |
| Homeostatic bugfix | `03-theory.md` §8 + Exp 02. |
| LM parity A2 (честный TF vs WFRLM) | Exp 09 + таблицы A100; узкий readout, sweep D, MLP при D=16 vs D=32 — в README Exp 09 и **H2** в §14. |
| Этап D (черновик) | Скрипт есть; примеры `longcontext_pair_*.json` в `experiments/09-lm-parity/outputs/` — **интерпретировать как черновик**, пока нет полного runbook в `remote_a100/runs/`. |

---

## 3. Поэтапный план реализации (следующий цикл)

Порядок согласован с [`12`](12-wfr-llm-breakthrough-roadmap.md) и [`14` §4](14-core-readiness-and-breakthrough-matrix.md).

### Этап P1 — Закрыть оставшийся инженерный хвост A/B (перед «магией RFP»)

1. **B3 combo** (если ещё не зафиксирован JSON в `runs/`):  
   `readout_wave_kernel=3` + MLP при **D=32**, тот же бюджет, что fair-parity — [`remote_gpu_b3_combo_d32.sh`](../experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh) в README Exp 09.
2. Повтор **H2** на **втором** сиде/бюджете (правило §14 §6): подтвердить, что MLP при D=16 вреден стабильно, при D=32 полезен стабильно.

### Этап P2 — C (масштаб train)

- Увеличить `PARITY_EPOCHS` / данные **после** фиксации лучшей связки P1.  
- Критерий: val CE продолжает падать у **обеих** моделей в сопоставимом режиме.

### Этап P3 — D (длинный контекст, U6)

1. Зафиксировать пару \(T_1, T_2\) (например 96 и 512) и правило для TF: полный attention vs скользящее окно — **явно в JSON**.
2. Прогон [`run_longcontext_pair.py`](../experiments/09-lm-parity/run_longcontext_pair.py) на A100 с `--fair-parity`, сохранить артефакт в `outputs/remote_a100/runs/…` по аналогии с sweep.
3. Отчёт: `best_val_ce`, `peak_cuda_mib`, `wall_seconds_full_pair` (уже в JSON).

### Этап P4 — B2 (опционально, U2)

- BPE / subword на том же корпусе с честным сравнением — отдельный скрипт или расширение загрузчика; **H8**.

### Этап P5 — E (RFP)

- Только после стабилизации P1–P3; критерии — [`12` этап E](12-wfr-llm-breakthrough-roadmap.md).

---

## 4. Реализация в репозитории (что сделано этим коммитом)

| Артефакт | Назначение |
|----------|------------|
| Этот файл (`docs/16-…`) | Единая точка: **проверка тезисов + фазы P1–P5** без расхождения с 12/14. |
| [`experiments/09-lm-parity/NEXT_PHASE_RUNBOOK.md`](../experiments/09-lm-parity/NEXT_PHASE_RUNBOOK.md) | Короткий runbook: команды копипастой для GPU (ссылки на существующие скрипты). |
| [`experiments/09-lm-parity/verify_longcontext_artifacts.py`](../experiments/09-lm-parity/verify_longcontext_artifacts.py) | Регрессия схемы JSON **D.1** после скачивания `longcontext_pair_*.json`. |
| [`experiments/09-lm-parity/run_local_pr_checklist.ps1`](../experiments/09-lm-parity/run_local_pr_checklist.ps1) | Локальный чеклист [`07-experiment-plan.md`](07-experiment-plan.md) (шаги 1–6 + verify); перед GPU по [`NEXT_PHASE_RUNBOOK.md`](../experiments/09-lm-parity/NEXT_PHASE_RUNBOOK.md). |

Обновляйте **§3 таблицу H*** в [`14`](14-core-readiness-and-breakthrough-matrix.md) после каждого нового JSON; этот документ — **навигация по смыслу**, а не дублирование цифр.

---

**Следующий документ по цепочке:** [`13-project-status-snapshot.md`](13-project-status-snapshot.md) для «живого» среза; матрица цифр — Exp 09 + §14.
