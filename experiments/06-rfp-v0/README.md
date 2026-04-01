# Experiment 06 — RFP v0 / v0.2

## Hypothesis

Plasticity rules on top of Adam and \(L = \alpha L_{\text{task}} + \beta(1-\text{RC}) + \gamma L_{\text{energy}}\) adjust `frequency`, `phase_bias`, and `decay` without replacing the main optimizer. **v0.2** (`rfp_step_v02`) adds a low-spike bonus to \(\Delta f\), softer decay updates, and \(\cos\) coupling between layers for \(\Delta\theta\).

## Parameters

- Toy next-token: `V=32`, `seq_len=48`, batch 16, Phase 0 Freq-Balanced.
- `--rfp-version v0 | v01 | v02` in `run_rfp_training.py`.
- v0.2 logs: `outputs/rfp_v02_log.json` (every 4 epochs by default).

## Success criteria

See [`docs/11-rfp-v0-spec.md`](../../docs/11-rfp-v0-spec.md). Artifacts: `outputs/ab_rfp_baseline.json`, `outputs/ab_rfp_v02.json`, `outputs/rfp_v02_grid.json`.

## How to run

```bash
# A/B all legacy modes
python experiments/06-rfp-v0/test_rfp_vs_baseline.py --quick --rfp-version all

# A/B baseline vs RFP v0.2 (50 epochs default without --quick)
python experiments/06-rfp-v0/test_rfp_vs_baseline.py --epochs 50 --rfp-version v02

# Single training run with logging
python experiments/06-rfp-v0/run_rfp_training.py --epochs 40 --rfp-version v02

# Hyperparameter grid (spike_target × eta_alpha × rfp_interval)
python experiments/06-rfp-v0/grid_rfp_v02.py --quick
```

## Verdict

**v0.2 implemented** — spike band and CE gains remain **run-dependent** on toy data; see spec §6–7 and grid summary for suggested hyperparameters.
