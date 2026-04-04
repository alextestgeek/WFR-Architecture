"""
Exp08 — сравнение схем входа (positions) на реальных данных.

Цель: проверить, что обучение next-token не зависит от «читерства» positions=tokens
и что wave-модель ведёт себя как ожидает теория при корректном positions=0..T-1.

Запуск:
  python experiments/08-wikitext-rfp/test_input_schemes.py --quick
  python experiments/08-wikitext-rfp/test_input_schemes.py --epochs 24
  python experiments/08-wikitext-rfp/test_input_schemes.py --long --minutes 45 --seeds 42,43
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(ROOT / "experiments" / "06-rfp-v0"))

from run_rfp_training import plot_ab_modes_bar  # noqa: E402
from run_wikitext_train import train_wikitext_run  # noqa: E402
from wikitext_loader import WikiTextCharCorpus  # noqa: E402


def plot_epoch_dynamics(
    epoch_log: list[dict],
    out_png: Path,
    *,
    title: str,
    vocab_size: int,
) -> None:
    """Динамика по эпохам: val CE, val acc@1, train CE, RC/spike."""
    if not epoch_log:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ln_v = math.log(vocab_size)
    ep = [float(e["epoch"]) for e in epoch_log]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(ep, [e["val_ce"] for e in epoch_log], color="#16a34a", label="val CE")
    axes[0].plot(ep, [e["best_val_ce_so_far"] for e in epoch_log], color="#15803d", alpha=0.6, linestyle="--", label="best val CE so far")
    axes[0].axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}")
    axes[0].set_ylabel("CE")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, [e["val_acc1"] for e in epoch_log], color="#7c3aed", label="val acc@1")
    axes[1].set_ylabel("acc@1")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, [e["train_ce_mean"] for e in epoch_log], color="#9333ea", label="train CE (epoch mean)")
    axes[2].set_ylabel("train CE")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(ep, [e["val_rc"] for e in epoch_log], color="#2563eb", label="val RC")
    axes[3].plot(ep, [e["spike_rate"] for e in epoch_log], color="#ea580c", label="spike rate")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("RC / spike")
    axes[3].legend(loc="best", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _artifact_stem(
    pos_mode: str,
    content_mode: str,
    content_scale: float,
    seed: int,
    ts: str,
) -> str:
    """Уникальные имена: absolute/normal при content_scale 1.0 и 0.5 иначе перезаписывали бы друг друга."""
    base = f"wikitext08_input_{pos_mode}_{content_mode}"
    if content_mode == "normal" and abs(content_scale - 1.0) > 1e-9:
        cs = str(content_scale).replace(".", "p")
        base = f"{base}_cs{cs}"
    return f"{base}_s{seed}_{ts}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--no-rfp", action="store_true")
    ap.add_argument("--rfp-version", type=str, default="v03", choices=("v02", "v03"))
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--minutes", type=int, default=0, help="длительность (прибл.): усиливает эпохи и батчи на GPU")
    ap.add_argument(
        "--long",
        action="store_true",
        help="длинный прогон GPU: floor --minutes>=40, seq до 128, больше эпох, *_dynamics.png; для 5 схем задайте --schemes full",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="init seeds list for stability check (comma-separated)",
    )
    ap.add_argument(
        "--schemes",
        type=str,
        default="core",
        choices=("core", "full"),
        help="core: 3 schemes; full: 5 schemes (includes content_scale/off)",
    )
    args = ap.parse_args()

    if args.long:
        # Нижняя граница длительности для «динамики по эпохам»; для 5 схем добавьте --schemes full
        args.minutes = max(args.minutes, 40)

    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=256)
    out_dir = _EXP / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.save_vocab_json(out_dir / "wikitext08_vocab_meta.json")

    epochs = 8 if args.quick else args.epochs
    n_train = 6 if args.quick else 12
    n_val = 3 if args.quick else 6
    seq_len = 96 if args.minutes else 64
    batch = 12
    save_png = not args.no_png
    save_dynamics = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No seeds parsed from --seeds")

    if args.schemes == "core":
        schemes = [
            ("absolute", "normal", 1.0),
            ("token_as_pos", "normal", 1.0),
            ("absolute", "off", 0.0),
        ]
    else:
        schemes = [
            ("absolute", "normal", 1.0),
            ("absolute_modV", "normal", 1.0),
            ("token_as_pos", "normal", 1.0),
            ("absolute", "normal", 0.5),
            ("absolute", "off", 0.0),
        ]

    # Важно: --minutes трактуем как общий бюджет на весь suite.
    # Поэтому scale считаем по "минуты на один прогон" (scheme × seed),
    # иначе 5 схем × 2-3 seeds превращаются в многократный сверхзапуск.
    if args.minutes and args.minutes > 0 and not args.quick:
        minutes_per_run = args.minutes / max(1, (len(seeds) * len(schemes)))

        if minutes_per_run >= 4:
            epochs = max(epochs, 30)
            n_train = max(n_train, 16)
            n_val = max(n_val, 8)
            seq_len = max(seq_len, 96)
        if minutes_per_run >= 6:
            epochs = max(epochs, 48)
            n_train = max(n_train, 18)
            n_val = max(n_val, 8)
            seq_len = max(seq_len, 112)
        if minutes_per_run >= 9:
            epochs = max(epochs, 72)
            n_train = max(n_train, 20)
            n_val = max(n_val, 8)
            seq_len = max(seq_len, 128)
        if minutes_per_run >= 12:
            epochs = max(epochs, 96)
            n_train = max(n_train, 24)
            n_val = max(n_val, 10)
            seq_len = max(seq_len, 128)

    per_run: list[dict[str, object]] = []
    # summary by scheme label
    summary: dict[str, dict[str, list[float]]] = {}

    for pos_mode, content_mode, content_scale in schemes:
        label = f"{pos_mode} / {content_mode}"
        if content_mode == "normal" and content_scale != 1.0:
            label += f" x{content_scale:g}"
        label += f" ({'baseline' if args.no_rfp else args.rfp_version})"
        summary[label] = {"best_val_ce": [], "final_val_acc1": []}

        for seed in seeds:
            stem = _artifact_stem(pos_mode, content_mode, content_scale, seed, ts)
            dyn_png = out_dir / f"{stem}_dynamics.png"
            m = train_wikitext_run(
                corpus,
                epochs=epochs,
                seq_len=seq_len,
                batch_size=batch,
                num_train_batches=n_train,
                num_val_batches=n_val,
                use_rfp=not args.no_rfp,
                rfp_interval=8,
                rfp_version=args.rfp_version,
                online_rfp=False,
                homeostatic_always_on=True,
                spike_rate_target=0.25,
                init_seed=seed,
                train_seed=42,
                val_seed=4242,
                lr=0.02,
                save_png=save_png,
                plot_path=out_dir / f"{stem}_curves.png" if save_png else None,
                epoch_json_path=out_dir / f"{stem}.json",
                verbose_epochs=bool(args.long or (args.minutes and args.minutes >= 40)),
                integrity_checks=True,
                manifest_path=out_dir / f"{stem}_manifest.json",
                epoch_csv_path=out_dir / f"{stem}.csv",
                pos_mode=pos_mode,
                content_mode=content_mode,
                content_scale=content_scale,
            )
            if save_dynamics and m.epoch_log:
                plot_epoch_dynamics(
                    m.epoch_log,
                    dyn_png,
                    title=f"{label} | seed={seed} | {epochs} ep",
                    vocab_size=corpus.vocab_size,
                )
            best_ce = float(m.metrics.best_val_ce)
            final_acc = float(m.epoch_log[-1]["val_acc1"]) if m.epoch_log else float("nan")
            per_run.append(
                {
                    "label": label,
                    "seed": seed,
                    "best_val_ce": best_ce,
                    "final_val_acc1": final_acc,
                    "dynamics_png": dyn_png.name if save_dynamics and dyn_png.is_file() else None,
                    "epoch_csv": f"{stem}.csv",
                }
            )
            summary[label]["best_val_ce"].append(best_ce)
            summary[label]["final_val_acc1"].append(final_acc)

    rows = []
    for label, vals in summary.items():
        ce = vals["best_val_ce"]
        acc = vals["final_val_acc1"]
        rows.append(
            {
                "mode": label,
                "best_val_ce_mean": statistics.mean(ce),
                "best_val_ce_std": statistics.pstdev(ce) if len(ce) > 1 else 0.0,
                "final_val_acc1_mean": statistics.mean(acc),
                "final_val_acc1_std": statistics.pstdev(acc) if len(acc) > 1 else 0.0,
            }
        )

    bar_png = out_dir / f"wikitext08_input_schemes_{ts}.png"
    if save_png:
        # bar chart still uses CE only
        plot_rows = [{"mode": r["mode"], "best_val_ce": float(r["best_val_ce_mean"])} for r in rows]
        plot_ab_modes_bar(plot_rows, bar_png, title=f"Exp08 input schemes (mean over seeds, {epochs} ep)", vocab_size=corpus.vocab_size)

    report = {
        "epochs": epochs,
        "seq_len": seq_len,
        "num_train_batches": n_train,
        "num_val_batches": n_val,
        "minutes": args.minutes,
        "long": args.long,
        "seeds": seeds,
        "schemes": args.schemes,
        "rows": rows,
        "per_run": per_run,
        "bar_png": bar_png.name if save_png else None,
        "dynamics_note": "Each run: *_dynamics.png (val CE, acc@1, train CE, RC/spike by epoch)",
    }
    out_json = out_dir / f"wikitext08_input_schemes_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()

