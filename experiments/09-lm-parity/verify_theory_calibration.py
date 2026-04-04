"""
Сверка следствий §9 `docs/03-theory.md` с артефактами parity JSON на диске.

Проверяет (локально, без GPU):
  H1 — монотонное сужение зазора Δ = CE(WFR)−CE(TF) при росте readout_feat_dim
       в «чистом» sweep (linear head, phase_causal=1, readout_wave=1, без MLP);
  B3 — при наличии трёх прогонов (контроль / phase_causal / readout_wave) согласованность
       с трактовкой H3/H4 (см. docs/13).

  python experiments/09-lm-parity/verify_theory_calibration.py
  python experiments/09-lm-parity/verify_theory_calibration.py path/to/runs/dir1 ...

Выход: текстовый отчёт; код возврата 0 если критические проверки пройдены, 1 если
       нет данных или нарушена монотонность H1 там, где есть ≥2 точек sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _flags(data: dict) -> tuple[int, int, int | None, int, int]:
    """readout_dim, phase_causal, mlp_hidden|None, readout_wave, neighbor as 0/1."""
    shared = data.get("shared") or {}
    cm = (data.get("fairness_notes") or {}).get("capacity_match") or {}
    rd = int(shared.get("readout_feat_dim") or cm.get("readout_feat_dim") or 0)
    pk = int(shared.get("phase_causal_kernel") or cm.get("phase_causal_kernel") or 1)
    rw = int(shared.get("readout_wave_kernel") or cm.get("readout_wave_kernel") or 1)
    mlp = shared.get("readout_mlp_hidden", cm.get("readout_mlp_hidden"))
    mlp_i = int(mlp) if mlp is not None else None
    nb = 1 if (shared.get("content_neighbor_mix") or cm.get("content_neighbor_mix")) else 0
    return rd, pk, mlp_i, rw, nb


def _delta(data: dict) -> float | None:
    comp = data.get("comparison") or {}
    d = comp.get("delta_best_val_ce_wfr_minus_transformer")
    return float(d) if d is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        help="каталоги с parity_pair_*.json (по умолчанию outputs/remote_a100/runs)",
    )
    args = ap.parse_args()
    base = Path(__file__).resolve().parent
    if args.dirs:
        roots = list(args.dirs)
    else:
        roots = [base / "outputs" / "remote_a100" / "runs"]

    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            print(f"skip (not a dir): {root}", file=sys.stderr)
            continue
        files.extend(sorted(root.rglob("parity_pair_*.json")))

    if not files:
        print("No parity_pair_*.json found.", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[Path, int, int, int | None, int, int, float | None]] = []
    for p in files:
        data = _load(p)
        if not data:
            continue
        rd, pk, mlp_i, rw, nb = _flags(data)
        dce = _delta(data)
        rows.append((p, rd, pk, mlp_i, rw, nb, dce))

    exit_code = 0

    # --- H1: clean sweep (linear, pk=1, rw=1, no neighbor, mlp None)
    clean = [
        r
        for r in rows
        if r[2] == 1 and r[3] is None and r[4] == 1 and r[5] == 0 and r[6] is not None
    ]
    sweep_only = [r for r in clean if "02_sweep" in str(r[0]).replace("\\", "/")]
    if len({r[1] for r in sweep_only}) >= 2:
        clean = sweep_only
    by_rd: dict[int, list[tuple[Path, float]]] = {}
    for p, rd, _, _, _, _, dce in clean:
        if rd < 3:
            continue
        by_rd.setdefault(rd, []).append((p, dce))

    print("=== H1 (readout sweep, linear head, no B3 flags) ===")
    if len(by_rd) < 2:
        print("  INSUFFICIENT_DATA: need ≥2 distinct readout_feat_dim with pk=1, rw=1, no MLP.")
        exit_code = 1
    else:
        sorted_ds = sorted(by_rd.keys())
        # Representative delta per D: min abs spread if multiple files
        rep: list[tuple[int, float, Path]] = []
        for rd in sorted_ds:
            best_p, best_d = min(by_rd[rd], key=lambda x: abs(x[1]))
            rep.append((rd, best_d, best_p))
        for rd, dce, p in rep:
            print(f"  D={rd:3d}  delta={dce:+.6f}  ({p.name})")
        # Monotonic narrowing: delta should not increase as D increases
        ok = True
        for i in range(1, len(rep)):
            if rep[i][1] > rep[i - 1][1] + 1e-9:
                ok = False
                print(
                    f"  FAIL: delta increased from D={rep[i - 1][0]} to D={rep[i][0]} "
                    f"({rep[i - 1][1]:+.6f} → {rep[i][1]:+.6f}) — contradicts H1 trend.",
                )
        if ok and len(rep) >= 2:
            print("  PASS: delta non-increasing with readout_feat_dim (gap narrows).")
        elif not ok:
            exit_code = 1

    # --- B3: same D, three ablations
    print()
    print("=== B3 (phase_causal vs readout_wave vs control), same readout ===")
    b3_candidates = [r for r in rows if r[1] == 16 and r[3] is None and r[5] == 0 and r[6] is not None]
    # group by (pk, rw)
    from collections import defaultdict

    grp: dict[tuple[int, int], list[tuple[Path, float]]] = defaultdict(list)
    for p, rd, pk, mlp_i, rw, nb, dce in b3_candidates:
        if mlp_i is not None:
            continue
        grp[(pk, rw)].append((p, dce))

    ctrl = grp.get((1, 1), [])
    pcausal = [(pk, rw, v) for (pk, rw), v in grp.items() if pk > 1 and rw == 1]
    rwave = [(pk, rw, v) for (pk, rw), v in grp.items() if rw > 1 and pk == 1]

    d0: float | None = None
    if not ctrl:
        print("  SKIP: no control (pk=1,rw=1) at D=16 in dataset.")
    else:
        d0 = min(ctrl, key=lambda x: abs(x[1]))[1]
        print(f"  control D=16 pk=1 rw=1: delta={d0:+.6f}")
    if pcausal:
        _, _, pairs = pcausal[0]
        d1 = min(pairs, key=lambda x: abs(x[1]))[1]
        parity_note = (
            abs(d1) < abs(d0) if d0 is not None else None
        )
        print(
            f"  phase_causal (pk>1): delta={d1:+.6f}  "
            f"(smaller |delta| than control: {parity_note})"
        )
    else:
        print("  (no phase_causal row at D=16)")
    if rwave:
        _, _, pairs = rwave[0]
        d2 = min(pairs, key=lambda x: abs(x[1]))[1]
        if d0 is not None:
            better = d2 < d0
            print(f"  readout_wave (rw>1): delta={d2:+.6f}  (narrower gap vs control: {better})")
        else:
            print(f"  readout_wave (rw>1): delta={d2:+.6f}")
    else:
        print("  (no readout_wave row at D=16)")

    print()
    print("See docs/03-theory.md section 9.2 and docs/14 section 3 for interpretation.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
