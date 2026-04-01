# 11. Resonant Field Plasticity (RFP) v0 — спецификация и статус

**Версия:** 1.0  
**Дата:** 1 апреля 2026  
**Статус:** реализовано в коде; критерии «primary» из плана требуют длинного обучения (30–50 эпох, GPU)

**Код:** корень репозитория — `wfr_lm.py`, `wfr_rfp.py`; ядро — `experiments/00-smoke-test/wfr_core.py`; эксперимент — [`experiments/06-rfp-v0/`](../experiments/06-rfp-v0/).

---

## 1. Цель

Перейти от sanity-check ([Experiment 05](../experiments/05-rfp-training-sanity/README.md), PASS) к **настоящему RFP v0**: правила \(\Delta f\), \(\Delta\theta\), \(\Delta\alpha\) накладываются **поверх** существующего Adam и композитного loss \(L = \alpha L_{\text{task}} + \beta(1-\text{RC}) + \gamma L_{\text{energy}}\), а не заменяют его.

---

## 2. Интерфейсы

- **`WFRState`** (`wfr_lm.py`): `resonance`, `spikes`, `rc`, `energy`, `phases`, `logits`, опционально `layer_resonance_means` (для `rfp_step_v01`).
- **`WFRLM`**: контентный канал `token_phase_offset` → `content_delta` в `WFRNetwork.forward`; readout `Linear` по признакам `[wave, RC, energy]` на каждом шаге последовательности.
- **`rfp_step` / `rfp_step_v01` / `apply_rfp_deltas`** (`wfr_rfp.py`): обновление `frequency`, `phase_bias`, `decay` с клипом; v0.1 добавляет к \(\Delta\theta\) член с \(\cos(\Delta)\) между соседними средними |резонанс| по слоям.

---

## 3. Изменения ядра (`wfr_core.py`)

| Функция | Описание |
|--------|----------|
| `TheoreticalResonanceLayer.phase_bias` | `nn.Parameter`, глобальный фазовый сдвиг \(\theta_l\) |
| `homeostatic_always_on` | при `True` homeostatic для порога спайка работает и в `train()` (для RFP v0) |
| `WFRNetwork.forward(..., content_delta=None)` | суммирование с фазами WPE после энкодера |
| `WFRNetwork.rc`, `WFRNetwork.threshold` | последний скалярный RC и средний порог по слоям |

RC — **скалярный** из `ResonanceConfidence`; спайки/energy в `WFRLM` по спецификации: жёсткий порог относительно среднего порога слоёв.

---

## 4. Обучение (Experiment 06)

Скрипт: `experiments/06-rfp-v0/run_rfp_training.py`  
A/B: `experiments/06-rfp-v0/test_rfp_vs_baseline.py`

Цикл: `total_loss.backward()` → `optimizer.step()` → при необходимости `rfp_step` + `apply_rfp_deltas` на `model.core` (каждые 8 шагов или online).

---

## 5. Результаты A/B (toy next-token, фиксированные батчи)

Прогон: `python experiments/06-rfp-v0/test_rfp_vs_baseline.py --epochs 24`, seed 42, `spike_rate_target=0.25`, homeostatic в train включён. Метрики — лучший val CE за прогон, финальный val RC, средняя доля спайков по `standing_wave` (порог), max L2 норма градиента.

| Режим | best val CE | Δ CE vs baseline | final val RC | spike rate | max ∥g∥₂ |
|-------|-------------|------------------|--------------|------------|----------|
| Adam only | 3.46709 | 0% | 0.978 | 0.040 | 0.189 |
| Adam + RFP v0 (каждые 8 шагов) | 3.46717 | −0.003% | ≈1.000 | 0.000 | 0.184 |
| Adam + RFP v0 (online) | 3.46721 | −0.004% | 0.999 | 0.000 | 0.180 |
| Adam + RFP v0.1 (каждые 8) | 3.46715 | −0.002% | 0.999 | 0.000 | 0.192 |

**Интерпретация:** на коротком toy-прогоне CE остаётся около \(\ln 32 \approx 3.47\); RFP пока не даёт заявленных −12% к CE. Часть режимов RFP даёт нулевую долю спайков на val (порог/standing wave после homeostatic+RFP). Для проверки **primary** критериев (RC ≥ 0.82 уже выполнен здесь; CE −12%; spike rate 0.18–0.32; ∥g∥ < 5) нужен **отдельный длинный прогон** с подбором \(\eta\), `clip`, `rfp_interval` и при необходимости `spike_rate_target`.

Артефакт JSON: `experiments/06-rfp-v0/outputs/ab_rfp_baseline.json`.

---

## 6. Критерии успеха (из плана)

**Primary (целевые):** val RC ≥ 0.82; val CE ниже baseline Adam ≥ 12%; spike rate 0.18–0.32; норма градиента < 5.0.

**Secondary:** online ≥ 1.4× быстрее по RC; корреляция обновлений `phase_bias` с улучшением RC > 0.6 (требуется логирование шагов).

---

## 7. Связанные документы

- [`03-theory.md`](03-theory.md) — §6, целевая функция  
- [`10-phase-1-plan.md`](10-phase-1-plan.md) — закрытие Phase 1, старт Phase 2  
- [`experiments/06-rfp-v0/`](../experiments/06-rfp-v0/) — скрипты и выходы
