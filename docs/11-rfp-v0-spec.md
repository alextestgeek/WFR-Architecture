# 11. Resonant Field Plasticity (RFP) v0 / v0.2 / v0.3 — спецификация и статус

**Версия документа:** 2.2 (RFP v0.3)  
**Дата:** 1 апреля 2026  
**Статус:** v0 реализован и A/B зафиксирован; **v0.2** — `rfp_step_v02` + grid; **v0.3** — `rfp_step_v03` (per-layer + rescue + мягкий homeostatic), `grid_rfp_v03.py`.

**Код:** корень — `wfr_lm.py`, `wfr_rfp.py`; ядро — `experiments/00-smoke-test/wfr_core.py`; эксперимент — [`experiments/06-rfp-v0/`](../experiments/06-rfp-v0/).

---

## 1. Цель

Перейти от sanity-check ([Experiment 05](../experiments/05-rfp-training-sanity/README.md), PASS) к **RFP** поверх Adam и композитного loss \(L = \alpha L_{\text{task}} + \beta(1-\text{RC}) + \gamma L_{\text{energy}}\).

**v0.2** устраняет типичные проблемы v0: **vanishing spikes** (нулевая доля спайков на val) и **слабый выигрыш по CE** — за счёт бонуса к частоте при низком \(r\), более мягкого \(\eta_\alpha\) на decay и логирования per-layer метрик.

---

## 2. Интерфейсы

- **`WFRState`** (`wfr_lm.py`): добавлены `layer_spike_means` (средняя доля спайков surrogate по слою), `layer_rc_share` (доля \(\lvert u_l\rvert\) в сумме по слоям — прокси «вклада» слоя при скалярном RC).
- **`rfp_step`** / **`rfp_step_v01`** / **`rfp_step_v02`** / **`rfp_step_v03`** / **`apply_rfp_deltas`** (`wfr_rfp.py`): v0.2 — `rfp_step_v02`; v0.3 — `rfp_step_v03` (формулы в docstring).

---

## 3. Изменения ядра (`wfr_core.py`)

Без обязательных новых правок для v0.2; используются существующие `phase_bias`, `content_delta`, `homeostatic_always_on`, скалярный RC.

---

## 4. Обучение и логи (Experiment 06)

| Скрипт | Назначение |
|--------|------------|
| `run_rfp_training.py` | `--rfp-version v0\|v01\|v02\|v03`; v02 → `outputs/rfp_v02_log.json`; v03 → `outputs/rfp_v03_log.json` (каждые `--log-every-epochs`, по умолчанию 4) |
| `test_rfp_vs_baseline.py` | `all` \| `v02` \| `v03` → `ab_rfp_baseline.json` / `ab_rfp_v02.json` / `ab_rfp_v03.json` |
| `grid_rfp_v02.py` | Сетка v0.2 → `outputs/rfp_v02_grid.json` |
| `grid_rfp_v03.py` | Сетка v0.3: `spike_rate_target` \(\in\{0.23,0.25,0.27\}\), `eta_alpha_v03` \(\in\{5\!\cdot\!10^{-6},8\!\cdot\!10^{-6},10^{-5}\}\), `rfp_interval` \(\in\{6,8,12\}\) → `outputs/rfp_v03_grid.json` |
| `run_long_v02_run022.py` | Длинный прогон лучшего v0.2 из grid (run 022): 80 эпох, `spike_target=0.28`, `eta_alpha_v02=3e-5`, `interval=8` |

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

### 6.3 RFP v0.3 (`rfp_step_v03`, `grid_rfp_v03.py`)

**Идея:** \(\Delta f_l\) и \(\Delta\theta_l\) используют **per-layer** `layer_rc_share` и `layer_spike_means`; к \(\Delta f_l\) добавлен **rescue** при низком \(r_l\); homeostatic по слою с \(\kappa_l\) (см. KaTeX в `wfr_rfp.py`).

**Команды:**

- Длинный v0.2 (лучший grid, run 022): `python experiments/06-rfp-v0/run_long_v02_run022.py` (80 эпох).
- Сетка v0.3: `python experiments/06-rfp-v0/grid_rfp_v03.py --epochs 50` (27 конфигов; `--quick` → 8 эпох).
- A/B baseline vs v0.3: `python experiments/06-rfp-v0/test_rfp_vs_baseline.py --epochs 50 --rfp-version v03`.

**Артефакты:** `outputs/rfp_v03_log.json`, `outputs/rfp_v03_grid.json`, `outputs/ab_rfp_v03.json`, кривые `outputs/rfp06_curves_*.png` (CE, RC, spike по эпохам).

| Прогон | best val CE | final val RC | mean spike | примечание |
|--------|-------------|--------------|------------|------------|
| Long v0.2 run022 (80 ep, GPU, 2026-04-01) | **3.46493** | 0.974 | **0.0** | CE чуть ниже ln 32; spike на val снова ушёл в 0 (см. `experiments/06-rfp-v0/outputs/remote_gpu/long_v02_run022_80ep_20260401_180846.json`) |
| Grid v0.3 (50 ep, GPU, 2026-04-01) | **3.46538** (лучший: `spike_target=0.25`, `η_α=5e-6`, `interval=8`) | до 1.0 | **0.0** (все 27) | Ни одна точка не попала в [0.18, 0.32]; `rescue_step_fraction=1.0` на всех шагах — rescue «всегда включён». Артефакт: `outputs/rfp_v03_grid.json` (копия: `outputs/remote_gpu/rfp_v03_grid.json`). |

---

## 7. Критерии успеха

### v0 (исторические primary)

val RC ≥ 0.82; val CE ниже baseline Adam ≥ 12%; spike rate 0.18–0.32; ∥g∥ < 5.

### v0.2 (целевые для анализа)

**Primary:** spike_rate на val **0.18–0.32** (не систематический ноль); val CE **ниже Adam ≥ 8%** на toy 50 эпох; **RC ≥ 0.85**; **max ∥g∥₂ < 3.0**.

**Secondary:** corr(обновления phase_bias, улучшение CE/RC) > 0.65; частоты/decay без взрыва; online v0.2 ≥ 1.3× скорость роста RC vs v0 (отдельный прогон).

### v0.3 (целевые)

**Primary:** mean spike на val **0.18–0.32**; val CE **ниже Adam ≥ 8%** (на toy 50 эпох — целевой порядок ≈ 3.18 при ln 32 ≈ 3.466); **final val RC ≥ 0.96**; **max ∥g∥₂ < 3.0**.

**Secondary:** Pearson `corr_delta_pb_delta_ce` ≥ 0.6; доля шагов с существенным rescue ≤ 20% (`rescue_step_fraction` в метриках и в `rfp_v03_log.json`).

---

## 8. Связанные документы

- [`03-theory.md`](03-theory.md) — §6  
- [`10-phase-1-plan.md`](10-phase-1-plan.md)  
- [`experiments/06-rfp-v0/README.md`](../experiments/06-rfp-v0/README.md)  
- [`experiments/06-rfp-protocol-tests/README.md`](../experiments/06-rfp-protocol-tests/README.md) — протокол **fresh-train + фиксированный val** и tier A/B/C (отдельно от toy grid)
