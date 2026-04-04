"""
Сводка по parity_pair_*.json: readout, CE, Δ, ratio параметров.

  python experiments/09-lm-parity/summarize_parity_json.py
  python experiments/09-lm-parity/summarize_parity_json.py path/to/run1 path/to/run2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_parity_dirs() -> list[Path]:
    base = Path(__file__).resolve().parent / "outputs" / "remote_a100"
    # Только полный sweep (без дублей readout 3/16 из отдельного прогона 01).
    return [base / "runs" / "02_sweep_readout_3_8_16_32_20260403"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        default=[],
        help="каталоги с parity_pair_*.json; по умолчанию только runs/02_sweep...",
    )
    args = ap.parse_args()
    dirs = list(args.dirs) if args.dirs else _default_parity_dirs()

    rows: list[tuple[str, ...]] = []
    for d in dirs:
        if not d.is_dir():
            print(f"skip (not a dir): {d}", file=sys.stderr)
            continue
        for p in sorted(d.glob("parity_pair_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"skip {p}: {e}", file=sys.stderr)
                continue
            shared = data.get("shared") or {}
            comp = data.get("comparison") or {}
            cm = (data.get("fairness_notes") or {}).get("capacity_match") or {}
            rd = shared.get("readout_feat_dim", cm.get("readout_feat_dim", "?"))
            mlp_h = shared.get("readout_mlp_hidden", cm.get("readout_mlp_hidden"))
            nb = shared.get("content_neighbor_mix", cm.get("content_neighbor_mix"))
            pk = shared.get("phase_causal_kernel", cm.get("phase_causal_kernel", 1))
            rw = shared.get("readout_wave_kernel", cm.get("readout_wave_kernel", 1))
            parts = [str(rd)]
            if mlp_h is not None:
                parts.append(f"mlp{mlp_h}")
            if nb:
                parts.append("nb")
            try:
                pk_i = int(pk)
            except (TypeError, ValueError):
                pk_i = 1
            if pk_i > 1:
                parts.append(f"pk{pk_i}")
            try:
                rw_i = int(rw)
            except (TypeError, ValueError):
                rw_i = 1
            if rw_i > 1:
                parts.append(f"rw{rw_i}")
            rd_cell = "+".join(parts)
            tf_ce = (data.get("transformer") or {}).get("best_val_ce")
            wfr_ce = (data.get("wfr_lm") or {}).get("best_val_ce")
            dce = comp.get("delta_best_val_ce_wfr_minus_transformer")
            ratio = comp.get("param_ratio_transformer_over_wfr")
            rows.append(
                (
                    p.name,
                    rd_cell,
                    f"{float(tf_ce):.4f}" if tf_ce is not None else "",
                    f"{float(wfr_ce):.4f}" if wfr_ce is not None else "",
                    f"{float(dce):+.4f}" if dce is not None else "",
                    f"{float(ratio):.4f}" if ratio is not None else "",
                )
            )

    if not rows:
        print("No parity_pair_*.json found.", file=sys.stderr)
        sys.exit(1)

    hdr = ("file", "readout", "CE_tf", "CE_wfr", "delta_wfr-tf", "param_ratio_tf/wfr")
    w = [max(len(hdr[i]), max(len(r[i]) for r in rows)) for i in range(6)]
    def line(cells: tuple[str, ...]) -> str:
        return " | ".join(cells[i].ljust(w[i]) for i in range(6))

    print(line(hdr))
    print("-+-".join("-" * w[i] for i in range(6)))
    for r in rows:
        print(line(r))


if __name__ == "__main__":
    main()
