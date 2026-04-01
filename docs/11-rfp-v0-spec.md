# 11. Resonant Field Plasticity (RFP) v0 / v0.2 — спецификация и статус

**Версия документа:** 2.1 (RFP v0.2)  
**Дата:** 1 апреля 2026  
**Статус:** v0 реализован и A/B зафиксирован; **v0.2** добавляет `rfp_step_v02`, расширенное логирование и grid гиперпараметров.

**Код:** корень — `wfr_lm.py`, `wfr_rfp.py`; ядро — `experiments/00-smoke-test/wfr_core.py`; эксперимент — [`experiments/06-rfp-v0/`](../experiments/06-rfp-v0/).

---

## 1. Цель

Перейти от sanity-check ([Experiment 05](../experiments/05-rfp-training-sanity/README.md), PASS) к **RFP** поверх Adam и композитного loss \(L = \alpha L_{\text{task}} + \beta(1-\text{RC}) + \gamma L_{\text{energy}}\).

**v0.2** устраняет типичные проблемы v0: **vanishing spikes** (нулевая доля спайков на val) и **слабый выигрыш по CE** — за счёт бонуса к частоте при низком \(r\), более мягкого \(\eta_\alpha\) на decay и логирования per-layer метрик.

---

## 2. Интерфейсы

- **`WFRState`** (`wfr_lm.py`): добавлены `layer_spike_means` (средняя доля спайков surrogate по слою), `layer_rc_share` (доля \(\lvert u_l\rvert\) в сумме по слоям — прокси «вклада» слоя при скалярном RC).
- **`rfp_step`** / **`rfp_step_v01`** / **`rfp_step_v02`** / **`apply_rfp_deltas`** (`wfr_rfp.py`): v0.2 — см. комментарии и KaTeX в `rfp_step_v02`.

---

## 3. Изменения ядра (`wfr_core.py`)

Без обязательных новых правок для v0.2; используются существующие `phase_bias`, `content_delta`, `homeostatic_always_on`, скалярный RC.

---

## 4. Обучение и логи (Experiment 06)

| Скрипт | Назначение |
|--------|------------|
| `run_rfp_training.py` | `--rfp-version v0\|v01\|v02`, при v02 — лог в `outputs/rfp_v02_log.json` (каждые `--log-every-epochs`, по умолчанию 4) |
| `test_rfp_vs_baseline.py` | `--rfp-version all` (как раньше) или `--rfp-version v02` → `ab_rfp_v02.json` |
| `grid_rfp_v02.py` | Сетка `spike_rate_target` \(\times\) `eta_alpha_v02` \(\times\) `rfp_interval` → `outputs/rfp_v02_grid.json` |

---

## 5. Результаты A/B — v0 (исторические)

Прогон: `test_rfp_vs_baseline.py --epochs 24`, seed 42. Артефакт: `outputs/ab_rfp_baseline.json`.

| Режим | best val CE | Δ CE vs baseline | final val RC | spike rate | max ∥g∥₂ |
|-------|-------------|------------------|--------------|------------|----------|
| Adam only | 3.46709 | 0% | 0.978 | 0.040 | 0.189 |
| Adam + RFP v0 (каждые 8) | 3.46717 | −0.003% | ≈1.000 | 0.000 | 0.184 |
| Adam + RFP v0 (online) | 3.46721 | −0.004% | 0.999 | 0.000 | 0.180 |
| Adam + RFP v0.1 (каждые 8) | 3.46715 | −0.002% | 0.999 | 0.000 | 0.192 |

---

## 6. Результаты A/B — v0.2

Команда по умолчанию для сравнения с baseline:  
`python experiments/06-rfp-v0/test_rfp_vs_baseline.py --epochs 50 --rfp-version v02`  
(короткий smoke: `--quick` даёт 16 эпох; без `--quick` по умолчанию 50 эпох для режима `v02`).

Артефакты: **`outputs/ab_rfp_v02.json`**, полный лог **`outputs/rfp_v02_log.json`** (per-layer spike_rate, per-layer rc_share, min/max frequency & decay, Pearson между суммой |Δphase_bias| за интервал и Δ val CE).

### 6.1 Пример строки A/B (24 эпохи, seed 42, default гиперы v0.2)

| Режим | best val CE | Δ CE vs baseline | final val RC | spike rate | max ∥g∥₂ | corr(d_pb, d_ce)* |
|-------|-------------|------------------|--------------|------------|----------|-------------------|
| Adam only | 3.46709 | 0% | 0.978 | 0.040 | 0.189 | — |
| Adam + RFP v0.2 (every 8) | 3.46711 | ≈0% | 0.991 | 0.000† | 0.190 | 0.335 |

\* Pearson по интервалам логирования (см. `rfp_v02_log.json`).  
† На этом прогоне mean spike по standing wave на val обнулился; на **коротком** прогоне `--quick --rfp-version v02` (16 эпох) для v0.2 наблюдался **spike_rate ≈ 0.26** в целевом диапазоне — метрика шумная из-за toy-данных и порога.

### 6.2 Grid гиперпараметров (v0.2, `grid_rfp_v02.py --quick`, 8 эпох на конфиг)

Сводка: **`outputs/rfp_v02_grid.json`**.

| Критерий | Лучший конфиг (по результатам grid) |
|----------|-------------------------------------|
| Минимальный val CE (все конфиги) | `spike_rate_target=0.22`, `eta_alpha_v02=5e-5`, `rfp_interval=16` → best val CE **3.46656**, spike 0 (короткий прогон) |
| **spike_rate ∈ [0.18, 0.32]** и лучший CE среди допустимых | `spike_rate_target=0.28`, `eta_alpha_v02=3e-5`, `rfp_interval=8` → mean_spike **0.191**, val CE **3.46740** |

Рекомендация для следующих длинных прогонов: стартовать с **spike_target=0.28**, **eta_alpha=3e-5**, **rfp_interval=8** при необходимости удержания доли спайков в коридоре.

---

## 7. Критерии успеха

### v0 (исторические primary)

val RC ≥ 0.82; val CE ниже baseline Adam ≥ 12%; spike rate 0.18–0.32; ∥g∥ < 5.

### v0.2 (целевые для анализа)

**Primary:** spike_rate на val **0.18–0.32** (не систематический ноль); val CE **ниже Adam ≥ 8%** на toy 50 эпох; **RC ≥ 0.85**; **max ∥g∥₂ < 3.0**.

**Secondary:** corr(обновления phase_bias, улучшение CE/RC) > 0.65; частоты/decay без взрыва; online v0.2 ≥ 1.3× скорость роста RC vs v0 (отдельный прогон).

---

## 8. Связанные документы

- [`03-theory.md`](03-theory.md) — §6  
- [`10-phase-1-plan.md`](10-phase-1-plan.md)  
- [`experiments/06-rfp-v0/README.md`](../experiments/06-rfp-v0/README.md)
