"""
A/B: Adam-only vs Adam+RFP (v0 и v0.1) на одной toy-задаче (Exp 06).

Запуск: python test_rfp_vs_baseline.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))

from run_rfp_training import train_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="короткий прогон для CI")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    epochs = 12 if args.quick else (args.epochs or 40)
    seed = 42

    rows = []
    baseline = train_run(
        epochs=epochs,
        use_rfp=False,
        rfp_interval=8,
        rfp_v01=False,
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        seed=seed,
    )
    rows.append(("Adam only", baseline))

    rfp_off = train_run(
        epochs=epochs,
        use_rfp=True,
        rfp_interval=8,
        rfp_v01=False,
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        seed=seed,
    )
    rows.append(("Adam + RFP v0 (every 8)", rfp_off))

    rfp_on = train_run(
        epochs=epochs,
        use_rfp=True,
        rfp_interval=8,
        rfp_v01=False,
        online_rfp=True,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        seed=seed,
    )
    rows.append(("Adam + RFP v0 (online)", rfp_on))

    rfp_v01 = train_run(
        epochs=epochs,
        use_rfp=True,
        rfp_interval=8,
        rfp_v01=True,
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        seed=seed,
    )
    rows.append(("Adam + RFP v0.1 (every 8)", rfp_v01))

    b_ce = baseline.best_val_ce
    table = []
    for name, m in rows:
        imp = (b_ce - m.best_val_ce) / b_ce * 100 if b_ce > 0 else 0.0
        table.append(
            {
                "mode": name,
                "best_val_ce": round(m.best_val_ce, 5),
                "val_ce_vs_baseline_pct": round(imp, 3),
                "final_val_rc": round(m.final_val_rc, 5),
                "spike_rate": round(m.mean_spike_rate, 5),
                "max_grad_l2": round(m.max_grad_l2, 5),
            }
        )

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "epochs": epochs,
        "baseline_best_val_ce": b_ce,
        "rows": table,
        "phase_bias_rc_correlation_proxy": None,
        "note": "Для полной метрики корреляции phase_bias vs RC нужна история шагов; см. длинный прогон.",
    }
    path = out_dir / "ab_rfp_baseline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
