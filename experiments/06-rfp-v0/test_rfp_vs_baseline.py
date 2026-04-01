"""
A/B: Adam-only vs Adam+RFP (v0, v0.1, v0.2) на toy next-token (Exp 06).

Примеры:
  python test_rfp_vs_baseline.py --quick
  python test_rfp_vs_baseline.py --epochs 50 --rfp-version v02
  python test_rfp_vs_baseline.py --epochs 80 --rfp-version v02   # без --quick
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
    ap.add_argument(
        "--rfp-version",
        type=str,
        choices=("all", "v02"),
        default="all",
        help="all: v0 / v0.1 / online; v02: baseline + RFP v0.2 только",
    )
    ap.add_argument("--spike-rate-target", type=float, default=0.25)
    ap.add_argument("--rfp-interval", type=int, default=8)
    ap.add_argument("--eta-alpha-v02", type=float, default=3e-5)
    args = ap.parse_args()

    if args.rfp_version == "v02":
        epochs = 16 if args.quick else (args.epochs or 50)
    else:
        epochs = 12 if args.quick else (args.epochs or 40)
    seed = 42

    rows: list[tuple[str, object]] = []
    log_v02_path = _EXP / "outputs" / "rfp_v02_log.json"

    if args.rfp_version == "all":
        baseline = train_run(
            epochs=epochs,
            use_rfp=False,
            rfp_interval=8,
            rfp_version="v0",
            online_rfp=False,
            homeostatic_always_on=True,
            spike_rate_target=0.25,
            seed=seed,
            log_path=None,
        )
        rows.append(("Adam only", baseline))

        rfp_off = train_run(
            epochs=epochs,
            use_rfp=True,
            rfp_interval=8,
            rfp_version="v0",
            online_rfp=False,
            homeostatic_always_on=True,
            spike_rate_target=0.25,
            seed=seed,
            log_path=None,
        )
        rows.append(("Adam + RFP v0 (every 8)", rfp_off))

        rfp_on = train_run(
            epochs=epochs,
            use_rfp=True,
            rfp_interval=8,
            rfp_version="v0",
            online_rfp=True,
            homeostatic_always_on=True,
            spike_rate_target=0.25,
            seed=seed,
            log_path=None,
        )
        rows.append(("Adam + RFP v0 (online)", rfp_on))

        rfp_v01 = train_run(
            epochs=epochs,
            use_rfp=True,
            rfp_interval=8,
            rfp_version="v01",
            online_rfp=False,
            homeostatic_always_on=True,
            spike_rate_target=0.25,
            seed=seed,
            log_path=None,
        )
        rows.append(("Adam + RFP v0.1 (every 8)", rfp_v01))
        out_name = "ab_rfp_baseline.json"
    else:
        baseline = train_run(
            epochs=epochs,
            use_rfp=False,
            rfp_interval=args.rfp_interval,
            rfp_version="v0",
            online_rfp=False,
            homeostatic_always_on=True,
            spike_rate_target=args.spike_rate_target,
            seed=seed,
            log_path=None,
        )
        rows.append(("Adam only", baseline))

        m_v02 = train_run(
            epochs=epochs,
            use_rfp=True,
            rfp_interval=args.rfp_interval,
            rfp_version="v02",
            online_rfp=False,
            homeostatic_always_on=True,
            spike_rate_target=args.spike_rate_target,
            seed=seed,
            log_path=log_v02_path,
            eta_alpha_v02=args.eta_alpha_v02,
        )
        rows.append((f"Adam + RFP v0.2 (every {args.rfp_interval})", m_v02))
        out_name = "ab_rfp_v02.json"

    b_ce = baseline.best_val_ce
    table = []
    for name, m in rows:
        imp = (b_ce - m.best_val_ce) / b_ce * 100 if b_ce > 0 else 0.0
        row = {
            "mode": name,
            "best_val_ce": round(m.best_val_ce, 5),
            "val_ce_vs_baseline_pct": round(imp, 3),
            "final_val_rc": round(m.final_val_rc, 5),
            "spike_rate": round(m.mean_spike_rate, 5),
            "max_grad_l2": round(m.max_grad_l2, 5),
        }
        if getattr(m, "corr_delta_pb_delta_ce", None) is not None:
            row["corr_delta_pb_delta_ce"] = (
                round(m.corr_delta_pb_delta_ce, 5)
                if m.corr_delta_pb_delta_ce is not None
                else None
            )
        table.append(row)

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    note = (
        "v02: corr_delta_pb_delta_ce = Pearson(d_pb, d_ce) over log intervals; see rfp_v02_log.json."
        if args.rfp_version == "v02"
        else "A/B toy next-token; v0.2 uses --rfp-version v02 and ab_rfp_v02.json."
    )
    report = {
        "epochs": epochs,
        "rfp_version_mode": args.rfp_version,
        "baseline_best_val_ce": b_ce,
        "rows": table,
        "phase_bias_rc_correlation_proxy": None,
        "note": note,
    }
    path = out_dir / out_name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {path}")
    if args.rfp_version == "v02":
        print(f"Full v0.2 log: {log_v02_path}")


if __name__ == "__main__":
    main()
