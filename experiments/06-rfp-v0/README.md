# Experiment 06 — RFP v0 / v0.2 / v0.3

## Hypothesis

Plasticity rules on top of Adam and \(L = \alpha L_{\text{task}} + \beta(1-\text{RC}) + \gamma L_{\text{energy}}\) adjust `frequency`, `phase_bias`, and `decay` without replacing the main optimizer. **v0.2** (`rfp_step_v02`) adds a low-spike bonus to \(\Delta f\), softer decay updates, and \(\cos\) coupling between layers for \(\Delta\theta\).

## Parameters

- Toy next-token: `V=32`, `seq_len=48`, batch 16, Phase 0 Freq-Balanced.
- `--rfp-version v0 | v01 | v02 | v03` in `run_rfp_training.py` / `test_rfp_vs_baseline.py` (`v03`: per-layer + rescue; лог по умолчанию `outputs/rfp_v03_log.json`).
- v0.2 logs: `outputs/rfp_v02_log.json` (every 4 epochs by default).

## Success criteria

See [`docs/11-rfp-v0-spec.md`](../../docs/11-rfp-v0-spec.md). Artifacts: `outputs/ab_rfp_baseline.json`, `outputs/ab_rfp_v02.json`, `outputs/rfp_v02_grid.json`.

## PNG (как в Exp 05)

После `test_rfp_vs_baseline.py` (по умолчанию):

- **`ab_rfp06_bar_<mode>_<timestamp>.png`** — столбцы best val CE по режимам, линия \(\ln V\).
- **`rfp06_curves_<mode>_<timestamp>.png`** — по одному файлу на каждый `train_run`: val CE, val RC, spike rate по эпохам.

Отключить: `--no-png`. Одиночный прогон: `run_rfp_training.py` пишет только кривые; флаг `--no-png` отключает их.

## How to run

```bash
# A/B all legacy modes
python experiments/06-rfp-v0/test_rfp_vs_baseline.py --quick --rfp-version all

# A/B baseline vs RFP v0.2 (50 epochs default without --quick)
python experiments/06-rfp-v0/test_rfp_vs_baseline.py --epochs 50 --rfp-version v02
# A/B baseline vs RFP v0.3
python experiments/06-rfp-v0/test_rfp_vs_baseline.py --epochs 50 --rfp-version v03

# Single training run with logging
python experiments/06-rfp-v0/run_rfp_training.py --epochs 40 --rfp-version v02
python experiments/06-rfp-v0/run_rfp_training.py --epochs 40 --rfp-version v03

# Hyperparameter grid (spike_target × eta_alpha × rfp_interval)
python experiments/06-rfp-v0/grid_rfp_v02.py --quick

# v0.3 grid (analogous knobs for v03)
python experiments/06-rfp-v0/grid_rfp_v03.py --quick
```

## Verdict

**v0.2 implemented** — spike band and CE gains remain **run-dependent** on toy data; see spec §6–7 and grid summary for suggested hyperparameters.
