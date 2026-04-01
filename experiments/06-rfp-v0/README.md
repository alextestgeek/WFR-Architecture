# Experiment 06 — RFP v0

## Hypothesis

Plasticity rules applied on top of Adam and the composite loss \(L = \alpha L_{\text{task}} + \beta(1-\text{RC}) + \gamma L_{\text{energy}}\) can adjust resonance parameters (`frequency`, `phase_bias`, `decay`) without replacing the main optimizer loop.

## Parameters

- Toy next-token: `V=32`, `seq_len=48`, batch 16, Phase 0 Freq-Balanced layer freqs/thresholds.
- Loss: same \(\alpha,\beta,\gamma\) as Experiment 05 defaults in `run_rfp_training.py`.
- RFP: `rfp_step` every 8 steps or online; optional `rfp_step_v01`; `homeostatic_always_on` on core for training.

## Success criteria

See [`docs/11-rfp-v0-spec.md`](../../docs/11-rfp-v0-spec.md) (primary/secondary targets). Short A/B metrics: `outputs/ab_rfp_baseline.json`.

## How to run

```bash
python experiments/06-rfp-v0/test_rfp_vs_baseline.py --quick
python experiments/06-rfp-v0/run_rfp_training.py --epochs 40 --out-json experiments/06-rfp-v0/outputs/last_run.json
```

## Verdict

**Requires follow-up (Phase 2)** — implementation complete; long runs and hyperparameter tuning are needed to meet primary CE/spike-rate goals.
