## Experiment 06 — LLM Trainability (content-conditioned WFR)

**Дата:** 31 марта 2026  
**Статус:** В работе

### Цель

Углублённо проверить «можно ли натренировать LLM» в смысле **next-token** на реальном тексте, а не на позиции:

- WFR получает **контент** через обучаемый `token_phase_offset` (эмбеддинг в пространство фаз) + позиционный WPE.
- Обучение идёт по CE, а также логируются компоненты §6: \((1-\overline{RC})\) и energy (доля surrogate-спайков).
- Метрика: val CE и perplexity; сравнение с baseline \(\ln V\).

### Почему это отдельный эксперимент

Experiment 05 использует `token_ids` как **positions** — это инженерный sanity градиентов, но **не** LLM (контент не входит в модель).

### Запуск (локально / GPU)

```bash
python experiments/06-llm-trainability/run_train_token_lm.py --epochs 30
python experiments/06-llm-trainability/run_train_token_lm.py --epochs 120 --strict
```

На удалённом GPU (по `RULES.md`): положить в плоскую папку `wfr_core.py` + `run_train_token_lm.py` + `phase0_best_config.py` (и запускать `.venv/bin/python`).

