"""
Tiered suite: baseline vs RFP (протокол fresh-train), отчёт JSON + проверки A/B/C.

Запуск с корня репозитория:
  python experiments/06-rfp-protocol-tests/run_tiered_suite.py --epochs 32

На удалённом GPU (см. README): выставить cwd на WFR-Memory-Test и PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(ROOT / "experiments" / "06-rfp-v0"))

from protocol_train import train_run_fresh_epochs  # noqa: E402
from tier_checks import check_tier_a_engineering, check_tier_b_field, check_tier_c_task_signal  # noqa: E402


def _tier_a_full(model, max_grad_l2: float, max_grad_cap: float = 25.0) -> dict:
    return check_tier_a_engineering(model, max_grad_l2, max_grad_cap=max_grad_cap)


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 06 protocol — tiered fresh-data suite")
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--quick", action="store_true", help="6 эпох — только smoke")
    p.add_argument("--rfp-interval", type=int, default=8)
    p.add_argument("--rfp-version", type=str, default="v03", choices=("v0", "v01", "v02", "v03"))
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=4242)
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    epochs = 6 if args.quick else args.epochs
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    common = dict(
        epochs=epochs,
        rfp_interval=args.rfp_interval,
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        num_train_batches=8,
        num_val_batches=4,
        save_png=not args.no_png,
    )

    base = train_run_fresh_epochs(
        use_rfp=False,
        rfp_version="v0",
        plot_title_prefix="baseline fresh",
        plot_path=out_dir / f"protocol06_baseline_{ts}.png" if not args.no_png else None,
        **common,
    )

    rfp = train_run_fresh_epochs(
        use_rfp=True,
        rfp_version=args.rfp_version,
        plot_title_prefix=f"RFP {args.rfp_version} fresh",
        plot_path=out_dir / f"protocol06_rfp_{args.rfp_version}_{ts}.png" if not args.no_png else None,
        log_path=out_dir / f"rfp_{args.rfp_version}_protocol_{ts}.json"
        if args.rfp_version in ("v02", "v03")
        else None,
        **common,
    )

    tier_a_b = _tier_a_full(base.model, base.metrics.max_grad_l2)
    tier_a_r = _tier_a_full(rfp.model, rfp.metrics.max_grad_l2)

    tier_b_b = check_tier_b_field(base.metrics.final_val_rc, base.metrics.mean_spike_rate)
    tier_b_r = check_tier_b_field(rfp.metrics.final_val_rc, rfp.metrics.mean_spike_rate)

    tier_c_b = check_tier_c_task_signal(base.history_val_ce)
    tier_c_r = check_tier_c_task_signal(rfp.history_val_ce)

    b_ce = base.metrics.best_val_ce
    r_ce = rfp.metrics.best_val_ce
    imp_pct = (b_ce - r_ce) / b_ce * 100 if b_ce > 0 else 0.0

    def _c_ok(t: dict) -> bool:
        if t.get("skipped"):
            return True
        return bool(t.get("pass"))

    report = {
        "suite": "06-rfp-protocol-tests",
        "protocol": "fresh_train_each_epoch_fixed_val_holdout",
        "epochs": epochs,
        "rfp_version": args.rfp_version,
        "baseline": {
            "metrics": asdict(base.metrics),
            "history_val_ce": base.history_val_ce,
            "tier_a": tier_a_b,
            "tier_b": tier_b_b,
            "tier_c": tier_c_b,
        },
        "rfp": {
            "metrics": asdict(rfp.metrics),
            "history_val_ce": rfp.history_val_ce,
            "tier_a": tier_a_r,
            "tier_b": tier_b_r,
            "tier_c": tier_c_r,
        },
        "comparison": {
            "best_val_ce_baseline": b_ce,
            "best_val_ce_rfp": r_ce,
            "rfp_vs_baseline_best_ce_pct": imp_pct,
        },
        "overall_pass": bool(
            tier_a_b.get("pass")
            and tier_a_r.get("pass")
            and tier_b_b.get("pass")
            and tier_b_r.get("pass")
            and _c_ok(tier_c_b)
            and _c_ok(tier_c_r)
        ),
    }

    out_path = args.out_json or (out_dir / f"tiered_suite_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
