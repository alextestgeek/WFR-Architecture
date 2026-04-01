"""
Длинный прогон лучшего конфига v0.2 из grid (run 022): spike_target=0.28, eta_alpha=3e-5, interval=8.

Запуск из корня:
  python experiments/06-rfp-v0/run_long_v02_run022.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))

from run_rfp_training import train_run  # noqa: E402


def main() -> None:
    out_dir = _EXP / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_p = out_dir / f"rfp_v02_long_run022_80ep_{ts}.json"
    m = train_run(
        epochs=80,
        use_rfp=True,
        rfp_interval=8,
        rfp_version="v02",
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.28,
        seed=42,
        log_path=log_p,
        eta_alpha_v02=3e-5,
    )
    summary = out_dir / f"long_v02_run022_80ep_{ts}.json"
    d = {
        "description": "v0.2 grid run 022 extended: spike_target=0.28, eta_alpha_v02=3e-5, rfp_interval=8, 80 epochs",
        "metrics": {
            "best_val_ce": m.best_val_ce,
            "final_val_rc": m.final_val_rc,
            "mean_spike_rate": m.mean_spike_rate,
            "max_grad_l2": m.max_grad_l2,
            "corr_delta_pb_delta_ce": m.corr_delta_pb_delta_ce,
            "mode": m.mode,
        },
        "detailed_log_path": str(log_p),
    }
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"\nWrote {summary}")


if __name__ == "__main__":
    main()
