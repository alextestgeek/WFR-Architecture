"""
A/B на WikiText-2 (char): Adam-only vs Adam+RFP v0.3 — как Exp 06, но реальные окна текста.

Запуск:
  python experiments/08-wikitext-rfp/test_wikitext_ab.py
  python experiments/08-wikitext-rfp/test_wikitext_ab.py --quick

PNG: ``outputs/wikitext08_ab_bar_*.png`` + по кривой на каждый режим ``wikitext08_curves_*``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(ROOT / "experiments" / "06-rfp-v0"))

from run_rfp_training import plot_ab_modes_bar  # noqa: E402
from run_wikitext_train import train_wikitext_run  # noqa: E402
from wikitext_loader import WikiTextCharCorpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="мало эпох/батчей")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--rfp-version", type=str, default="v03", choices=("v02", "v03"))
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=256)
    out_dir = _EXP / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.save_vocab_json(out_dir / "wikitext08_vocab_meta.json")

    epochs = 6 if args.quick else args.epochs
    n_train = 4 if args.quick else 6
    n_val = 2 if args.quick else 4
    seq_len = 48 if args.quick else 64
    batch = 8 if args.quick else 12

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_png = not args.no_png

    base = train_wikitext_run(
        corpus,
        epochs=epochs,
        seq_len=seq_len,
        batch_size=batch,
        num_train_batches=n_train,
        num_val_batches=n_val,
        use_rfp=False,
        rfp_interval=8,
        rfp_version="v0",
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        save_png=save_png,
        plot_path=out_dir / f"wikitext08_curves_baseline_{ts}.png" if save_png else None,
        epoch_json_path=out_dir / f"wikitext08_epoch_baseline_{ts}.json",
    )

    rfp = train_wikitext_run(
        corpus,
        epochs=epochs,
        seq_len=seq_len,
        batch_size=batch,
        num_train_batches=n_train,
        num_val_batches=n_val,
        use_rfp=True,
        rfp_interval=8,
        rfp_version=args.rfp_version,
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        save_png=save_png,
        plot_path=out_dir / f"wikitext08_curves_rfp_{args.rfp_version}_{ts}.png" if save_png else None,
        epoch_json_path=out_dir / f"wikitext08_epoch_rfp_{args.rfp_version}_{ts}.json",
        log_path=out_dir / f"wikitext08_rfp_{args.rfp_version}_detail_{ts}.json",
    )

    b_ce = base.metrics.best_val_ce
    rows_chart = [
        {"mode": "Adam only (WikiText char)", "best_val_ce": float(base.metrics.best_val_ce)},
        {
            "mode": f"Adam + RFP {args.rfp_version} (WikiText char)",
            "best_val_ce": float(rfp.metrics.best_val_ce),
        },
    ]
    report = {
        "epochs": epochs,
        "seq_len": seq_len,
        "batch_size": batch,
        "vocab_size": corpus.vocab_size,
        "baseline_best_val_ce": b_ce,
        "rows": rows_chart,
        "metrics_baseline": asdict(base.metrics),
        "metrics_rfp": asdict(rfp.metrics),
        "artifacts": {
            "ab_bar_png": f"wikitext08_ab_bar_{ts}.png" if save_png else None,
            "curves_baseline": f"wikitext08_curves_baseline_{ts}.png" if save_png else None,
            "curves_rfp": f"wikitext08_curves_rfp_{args.rfp_version}_{ts}.png" if save_png else None,
        },
    }
    path = out_dir / f"wikitext08_ab_{args.rfp_version}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    if save_png:
        plot_ab_modes_bar(
            rows_chart,
            out_dir / f"wikitext08_ab_bar_{ts}.png",
            title=f"Experiment 08 WikiText — best val CE ({epochs} ep, char)",
            vocab_size=corpus.vocab_size,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
