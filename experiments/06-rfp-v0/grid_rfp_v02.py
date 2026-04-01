"""
Малый grid по гиперпараметрам RFP v0.2 (spike_rate_target × eta_alpha × rfp_interval).

Запуск из корня:
  python experiments/06-rfp-v0/grid_rfp_v02.py --epochs 24 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))

from run_rfp_training import train_run  # noqa: E402


def in_spike_band(r: float) -> bool:
    return 0.18 <= r <= 0.32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--quick", action="store_true", help="уменьшить epochs до 8")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    epochs = 8 if args.quick else args.epochs

    spike_targets = [0.22, 0.25, 0.28]
    eta_alphas = [1e-5, 3e-5, 5e-5]
    intervals = [4, 8, 16]

    results = []
    best_ce = float("inf")
    best_row = None

    for idx, (srt, ea, ri) in enumerate(product(spike_targets, eta_alphas, intervals)):
        log_p = _EXP / "outputs" / f"rfp_v02_grid_run_{idx:03d}.json"
        m = train_run(
            epochs=epochs,
            use_rfp=True,
            rfp_interval=ri,
            rfp_version="v02",
            online_rfp=False,
            homeostatic_always_on=True,
            spike_rate_target=srt,
            seed=args.seed,
            log_path=log_p,
            eta_alpha_v02=ea,
        )
        row = {
            "spike_rate_target": srt,
            "eta_alpha_v02": ea,
            "rfp_interval": ri,
            "best_val_ce": m.best_val_ce,
            "final_val_rc": m.final_val_rc,
            "mean_spike_rate": m.mean_spike_rate,
            "max_grad_l2": m.max_grad_l2,
            "corr_delta_pb_delta_ce": m.corr_delta_pb_delta_ce,
            "spike_in_0_18_0_32": in_spike_band(m.mean_spike_rate),
            "log_path": str(log_p),
        }
        results.append(row)
        if m.best_val_ce < best_ce:
            best_ce = m.best_val_ce
            best_row = row

    in_band = [r for r in results if r["spike_in_0_18_0_32"]]
    best_in_band = None
    if in_band:
        best_in_band = min(in_band, key=lambda x: x["best_val_ce"])

    out_path = _EXP / "outputs" / "rfp_v02_grid.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epochs": epochs,
        "grid": {"spike_rate_target": spike_targets, "eta_alpha_v02": eta_alphas, "rfp_interval": intervals},
        "best_val_ce_overall": best_row,
        "best_in_spike_band_0_18_0_32": best_in_band,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
