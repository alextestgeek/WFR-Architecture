"""
Этап D (черновик D.1): два (или несколько) seq_len подряд — те же fair-parity правила,
для каждой длины отдельный match-capacity; отчёт CE + wall time + peak CUDA memory.

Запуск из корня:
  python experiments/09-lm-parity/run_longcontext_pair.py --fair-parity --seq-lens 96,512 \\
    --epochs 48 --num-train-batches 20 --num-val-batches 8 --batch-size 16

С `--quick` эпохи и батчи ужимаются как в run_parity_pair.

Выход: experiments/09-lm-parity/outputs/longcontext_pair_<ts>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_EXP08 = ROOT / "experiments" / "08-wikitext-rfp"
_EXP09 = Path(__file__).resolve().parent
_EXP06 = ROOT / "experiments" / "06-rfp-v0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_EXP08))
sys.path.insert(0, str(_EXP06))
sys.path.insert(0, str(_EXP09))

import torch  # noqa: E402

from run_parity_pair import (  # noqa: E402
    build_parity_argument_parser,
    normalize_parity_args,
    run_parity_train_payload,
)
from wikitext_loader import WikiTextCharCorpus  # noqa: E402


def _parse_seq_lens(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if len(out) < 1:
        raise SystemExit("--seq-lens needs at least one integer")
    return out


def main() -> None:
    ap = build_parity_argument_parser()
    ap.add_argument(
        "--seq-lens",
        type=str,
        default="96,512",
        help="через запятую, например 96,512 — отдельный parity-прогон на каждой длине",
    )
    args0 = ap.parse_args()
    seq_lens = _parse_seq_lens(args0.seq_lens)

    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=256)
    runs: list[dict] = []

    for seq_len in seq_lens:
        args = copy.deepcopy(args0)
        args.seq_len = seq_len
        wfr_lr = normalize_parity_args(args)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        payload = run_parity_train_payload(args, wfr_lr, corpus)
        t1 = time.perf_counter()

        peak_mib = None
        if torch.cuda.is_available():
            peak_mib = round(torch.cuda.max_memory_allocated() / (1024**2), 3)

        comp = payload.get("comparison") or {}
        row = {
            "seq_len": seq_len,
            "parity_payload": payload,
            "d1_metrics": {
                "wall_seconds_full_pair": round(t1 - t0, 3),
                "peak_cuda_mib": peak_mib,
                "cuda_available": torch.cuda.is_available(),
            },
        }
        runs.append(row)
        dce = comp.get("delta_best_val_ce_wfr_minus_transformer")
        print(f"seq_len={seq_len} delta={dce} wall_s={row['d1_metrics']['wall_seconds_full_pair']} peak_mib={peak_mib}")

    out = {
        "experiment": "09-lm-parity-D1-draft",
        "roadmap": "docs/12-wfr-llm-breakthrough-roadmap.md §D.1",
        "seq_lens_requested": seq_lens,
        "runs": runs,
    }
    out_dir = _EXP09 / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"longcontext_pair_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
