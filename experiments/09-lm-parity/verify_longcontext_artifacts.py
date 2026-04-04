"""
Проверка структуры JSON этапа D.1 (run_longcontext_pair.py).

Ищет longcontext_pair_*.json, проверяет обязательные поля и печатает сводку по Δ,
peak CUDA и wall time — без GPU.

  python experiments/09-lm-parity/verify_longcontext_artifacts.py
  python experiments/09-lm-parity/verify_longcontext_artifacts.py path/to/dir

Код возврата: 0 если найден хотя бы один валидный файл; 1 если нет файлов или все с ошибками.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ERROR: {p.name}: {e}", file=sys.stderr)
        return None


def _validate_and_summarize(data: dict, path: Path) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    if data.get("experiment") != "09-lm-parity-D1-draft":
        lines.append(f"  WARN: experiment={data.get('experiment')!r} (expected 09-lm-parity-D1-draft)")
    runs = data.get("runs")
    if not isinstance(runs, list) or len(runs) < 1:
        lines.append("  FAIL: missing or empty runs[]")
        return False, lines

    seq_req = data.get("seq_lens_requested")
    if isinstance(seq_req, list):
        lines.append(f"  seq_lens_requested: {seq_req}")

    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            ok = False
            lines.append(f"  FAIL: runs[{i}] not an object")
            continue
        sl = run.get("seq_len")
        pl = run.get("parity_payload")
        m = run.get("d1_metrics")
        if sl is None:
            ok = False
            lines.append(f"  FAIL: runs[{i}] missing seq_len")
        if not isinstance(pl, dict):
            ok = False
            lines.append(f"  FAIL: runs[{i}] missing parity_payload")
        else:
            comp = pl.get("comparison") or {}
            dce = comp.get("delta_best_val_ce_wfr_minus_transformer")
            tf = (pl.get("transformer") or {}).get("best_val_ce")
            wf = (pl.get("wfr_lm") or {}).get("best_val_ce")
            sh = (pl.get("shared") or {})
            rd = sh.get("readout_feat_dim", "?")
            ep = sh.get("epochs", "?")
            lines.append(
                f"    seq_len={sl}  readout={rd}  epochs={ep}  "
                f"CE_tf={tf}  CE_wfr={wf}  delta={dce}"
            )
        if not isinstance(m, dict):
            ok = False
            lines.append(f"  FAIL: runs[{i}] missing d1_metrics")
        else:
            lines.append(
                f"    d1: wall_s={m.get('wall_seconds_full_pair')}  "
                f"peak_cuda_mib={m.get('peak_cuda_mib')}  cuda={m.get('cuda_available')}"
            )

    # Multi-length: informational trend
    if len(runs) >= 2:
        deltas: list[tuple[int, float | None]] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            pl = run.get("parity_payload") or {}
            comp = pl.get("comparison") or {}
            d = comp.get("delta_best_val_ce_wfr_minus_transformer")
            sl = run.get("seq_len")
            if isinstance(sl, int) and d is not None:
                deltas.append((sl, float(d)))
        deltas.sort(key=lambda x: x[0])
        if len(deltas) >= 2:
            lines.append(f"  NOTE: delta vs seq_len trend: {deltas}")

    return ok, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="корни поиска (по умолчанию experiments/09-lm-parity/outputs)",
    )
    args = ap.parse_args()
    base = Path(__file__).resolve().parent
    roots = list(args.roots) if args.roots else [base / "outputs"]

    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.name.startswith("longcontext_pair_") and root.suffix == ".json":
            files.append(root)
            continue
        if root.is_dir():
            files.extend(sorted(root.rglob("longcontext_pair_*.json")))
        else:
            print(f"skip (not found): {root}", file=sys.stderr)

    files = sorted(set(files))
    if not files:
        print("No longcontext_pair_*.json found.", file=sys.stderr)
        sys.exit(1)

    any_ok = False
    print("=== D.1 longcontext_pair artifacts ===")
    for p in files:
        data = _load(p)
        if not data:
            continue
        ok, lines = _validate_and_summarize(data, p)
        status = "PASS" if ok else "FAIL"
        if ok:
            any_ok = True
        try:
            disp = p.relative_to(base)
        except ValueError:
            disp = p
        print(f"  [{status}] {disp}")
        for line in lines:
            print(line)

    if not any_ok:
        print("No valid longcontext artifacts (schema check failed).", file=sys.stderr)
        sys.exit(1)

    print()
    print("See docs/12-wfr-llm-breakthrough-roadmap.md section D.1 (protocol).")
    sys.exit(0)


if __name__ == "__main__":
    main()
