"""
Exp 09 — этап A2: парный прогон Transformer vs WFRLM на одном протоколе WikiText char.

Условия честности:
  - Один корпус, одни val-батчи (val_seed=4242), fresh train по эпохам.
  - WFRLM: без RFP, loss_mode=ce_only — та же цель обучения, что у Transformer (только CE).

Дорожная карта: docs/12-wfr-llm-breakthrough-roadmap.md §A2.

Запуск из корня:
  python experiments/09-lm-parity/run_parity_pair.py --quick
  python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity
  python experiments/09-lm-parity/run_parity_pair.py --epochs 24 --num-train-batches 10 --seq-len 96

Импорт для этапа D: run_parity_train_payload(args, wfr_lr) → dict
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

from parity_capacity import (  # noqa: E402
    count_wfr_lm_trainable,
    match_transformer_to_wfr_params,
)
from run_transformer_char_baseline import train_run as transformer_train_run  # noqa: E402
from run_wikitext_train import train_wikitext_run  # noqa: E402
from wikitext_loader import WikiTextCharCorpus  # noqa: E402


def build_parity_argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Exp 09 A2: Transformer vs WFRLM parity pair")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-train-batches", type=int, default=6)
    ap.add_argument("--num-val-batches", type=int, default=4)
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--val-seed", type=int, default=4242)
    ap.add_argument("--num-resonance-layers", type=int, default=4, metavar="N")
    ap.add_argument(
        "--readout-feat-dim",
        type=int,
        default=3,
        metavar="D",
        help="WFRLM readout (дорожная карта B1); при --match-capacity пересчитывается подбор TF",
    )
    ap.add_argument(
        "--readout-mlp-hidden",
        type=int,
        default=None,
        metavar="H",
        help="MLP readout D→H→vocab; при --match-capacity учитывается в числе параметров WFR",
    )
    ap.add_argument(
        "--content-neighbor-mix",
        action="store_true",
        help="WFRLM: причинное смешивание δ[t−1] в content_delta (§9.1 теории; +1 trainable scalar)",
    )
    ap.add_argument(
        "--phase-causal-kernel",
        type=int,
        default=1,
        metavar="K",
        help="WFRNetwork: каузальная depthwise смесь фаз по времени (K>1); учитывается в --match-capacity",
    )
    ap.add_argument(
        "--readout-wave-kernel",
        type=int,
        default=1,
        metavar="K",
        help="WFRLM: каузальная смесь standing_wave перед readout (K>1); учитывается в --match-capacity",
    )
    ap.add_argument(
        "--fair-parity",
        action="store_true",
        help="как --match-capacity --match-optimizer (общий рецепт оптимизации + подбор параметров TF)",
    )
    ap.add_argument("--tf-lr", type=float, default=3e-4, help="LR TinyCausalTransformer (AdamW)")
    ap.add_argument("--wfr-lr", type=float, default=0.02, help="LR WFRLM (Adam, как Exp 08)")
    ap.add_argument("--match-lr", action="store_true", help="оба optim на --tf-lr (жёстче сопоставить масштаб шага)")
    ap.add_argument(
        "--match-optimizer",
        action="store_true",
        help="WFRLM: AdamW + grad clip 1.0 как у TinyCausalTransformer; LR для WFR = --tf-lr",
    )
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--dim-ff", type=int, default=512)
    ap.add_argument(
        "--match-capacity",
        action="store_true",
        help="подобрать d_model/nhead/nlayers/dim-ff Transformer ≈ числу параметров WFRLM (--num-resonance-layers, vocab)",
    )
    ap.add_argument(
        "--match-capacity-min-d-model",
        type=int,
        default=8,
        metavar="D",
        help="нижняя граница d_model при --match-capacity (8 по умолчанию; 4 допускает минимальный |Δn|)",
    )
    return ap


def normalize_parity_args(args: argparse.Namespace) -> float:
    """Валидация и мутация args (fair-parity, quick); возвращает LR для WFRLM."""
    if args.fair_parity:
        args.match_capacity = True
        args.match_optimizer = True
    mlp_h = args.readout_mlp_hidden
    if args.readout_feat_dim < 3:
        raise SystemExit("--readout-feat-dim must be >= 3")
    if mlp_h is not None and mlp_h < 1:
        raise SystemExit("--readout-mlp-hidden must be >= 1 when set")
    if args.phase_causal_kernel < 1:
        raise SystemExit("--phase-causal-kernel must be >= 1")
    if args.readout_wave_kernel < 1:
        raise SystemExit("--readout-wave-kernel must be >= 1")

    if args.quick:
        args.epochs = min(args.epochs, 6)
        args.num_train_batches = min(args.num_train_batches, 4)
        args.num_val_batches = min(args.num_val_batches, 2)
        args.batch_size = min(args.batch_size, 8)

    if args.match_optimizer:
        return args.tf_lr
    return args.tf_lr if args.match_lr else args.wfr_lr


def run_parity_train_payload(args: argparse.Namespace, wfr_lr: float, corpus: WikiTextCharCorpus) -> dict:
    """
    Один полный парный прогон. args уже нормализован normalize_parity_args.
    Мутирует args при match_capacity (d_model, ...).
    """
    mlp_h = args.readout_mlp_hidden
    out_dir = _EXP09 / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity_match_meta: dict | None = None
    if args.match_capacity:
        wfr_target_n = count_wfr_lm_trainable(
            corpus.vocab_size,
            args.num_resonance_layers,
            spike_rate_target=0.25,
            readout_feat_dim=args.readout_feat_dim,
            readout_mlp_hidden=mlp_h,
            content_neighbor_mix=args.content_neighbor_mix,
            phase_causal_kernel=args.phase_causal_kernel,
            readout_wave_kernel=args.readout_wave_kernel,
        )
        matched = match_transformer_to_wfr_params(
            wfr_target_n,
            corpus.vocab_size,
            args.seq_len,
            min_d_model=args.match_capacity_min_d_model,
        )
        args.d_model = matched.d_model
        args.nhead = matched.nhead
        args.nlayers = matched.nlayers
        args.dim_ff = matched.dim_feedforward
        capacity_match_meta = {
            "enabled": True,
            "readout_feat_dim": args.readout_feat_dim,
            "readout_mlp_hidden": mlp_h,
            "content_neighbor_mix": args.content_neighbor_mix,
            "phase_causal_kernel": args.phase_causal_kernel,
            "readout_wave_kernel": args.readout_wave_kernel,
            "min_d_model": args.match_capacity_min_d_model,
            "wfr_trainable_params_target": wfr_target_n,
            "transformer_trainable_params": matched.num_params,
            "param_ratio_transformer_over_wfr": matched.param_ratio_tf_over_wfr,
            "chosen": {
                "d_model": matched.d_model,
                "nhead": matched.nhead,
                "nlayers": matched.nlayers,
                "dim_feedforward": matched.dim_feedforward,
            },
        }

    br, tf_log = transformer_train_run(
        corpus,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_train_batches=args.num_train_batches,
        num_val_batches=args.num_val_batches,
        lr=args.tf_lr,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dim_feedforward=args.dim_ff,
        init_seed=args.init_seed,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
    )

    wfr = train_wikitext_run(
        corpus,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_train_batches=args.num_train_batches,
        num_val_batches=args.num_val_batches,
        use_rfp=False,
        rfp_interval=8,
        rfp_version="v03",
        online_rfp=False,
        homeostatic_always_on=True,
        spike_rate_target=0.25,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        init_seed=args.init_seed,
        lr=wfr_lr,
        log_path=None,
        save_png=False,
        plot_path=None,
        epoch_json_path=None,
        verbose_epochs=False,
        integrity_checks=True,
        manifest_path=None,
        epoch_csv_path=None,
        pos_mode="absolute",
        content_mode="normal",
        content_scale=1.0,
        num_resonance_layers=args.num_resonance_layers,
        loss_mode="ce_only",
        readout_feat_dim=args.readout_feat_dim,
        readout_mlp_hidden=mlp_h,
        content_neighbor_mix=args.content_neighbor_mix,
        phase_causal_kernel=args.phase_causal_kernel,
        readout_wave_kernel=args.readout_wave_kernel,
        optimizer="adamw" if args.match_optimizer else "adam",
        grad_clip_norm=1.0 if args.match_optimizer else None,
    )

    tf_ce = float(br.best_val_ce)
    wfr_ce = float(wfr.metrics.best_val_ce)
    wfr_n = int(wfr.manifest.get("num_trainable_params") or 0)
    tf_n = int(br.num_params)
    cap_ratio = (tf_n / max(1, wfr_n)) if wfr_n else None
    return {
        "experiment": "09-lm-parity-A2",
        "roadmap": "docs/12-wfr-llm-breakthrough-roadmap.md section A2",
        "shared": {
            "epochs": args.epochs,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "num_train_batches": args.num_train_batches,
            "num_val_batches": args.num_val_batches,
            "init_seed": args.init_seed,
            "train_seed": args.train_seed,
            "val_seed": args.val_seed,
            "readout_feat_dim": args.readout_feat_dim,
            "readout_mlp_hidden": mlp_h,
            "content_neighbor_mix": args.content_neighbor_mix,
            "phase_causal_kernel": args.phase_causal_kernel,
            "readout_wave_kernel": args.readout_wave_kernel,
            "corpus_val_sha256_fingerprint": wfr.val_fingerprint,
        },
        "fairness_notes": {
            "wfr_objective": "ce_only, no RFP (matches next-token CE only)",
            "transformer_objective": "cross_entropy only",
            "optimizer": (
                "WFRLM: AdamW (betas 0.9/0.95, wd 0.1) + clip 1.0; Transformer: то же; общий LR (--tf-lr)"
                if args.match_optimizer
                else "WFRLM: Adam; Transformer: AdamW + grad clip — см. скрипты"
            ),
            "lr_transformer": args.tf_lr,
            "lr_wfr_lm": wfr_lr,
            "match_lr_flag": args.match_lr,
            "match_optimizer_flag": args.match_optimizer,
            "capacity_match": capacity_match_meta,
        },
        "transformer": {
            **asdict(br),
            "epoch_log_tail": tf_log[-3:] if len(tf_log) > 3 else tf_log,
        },
        "wfr_lm": {
            "best_val_ce": wfr.metrics.best_val_ce,
            "final_val_rc": wfr.metrics.final_val_rc,
            "mean_spike_rate_end": wfr.metrics.mean_spike_rate,
            "mode": wfr.metrics.mode,
            "num_trainable_params": wfr.manifest.get("num_trainable_params"),
            "max_grad_l2": wfr.metrics.max_grad_l2,
        },
        "comparison": {
            "delta_best_val_ce_wfr_minus_transformer": wfr_ce - tf_ce,
            "trainable_params_transformer": tf_n,
            "trainable_params_wfr_lm": wfr_n,
            "param_ratio_transformer_over_wfr": cap_ratio,
            "interpretation": "положительно → WFR хуже baseline TF по best val CE при этом протоколе; "
            "отрицательно → WFR лучше (редко при первых прогонах).",
            "capacity_warning": (
                "При param_ratio ≫ 1 сравнение CE в первую очередь отражает разную ёмкость. "
                "Уменьшите --d-model/--nlayers/--dim-ff у Transformer или расширяйте WFRLM (дорожная карта §B)."
                if cap_ratio is not None and cap_ratio > 3.0
                else None
            ),
        },
    }


def main() -> None:
    ap = build_parity_argument_parser()
    args = ap.parse_args()
    wfr_lr = normalize_parity_args(args)
    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=256)
    payload = run_parity_train_payload(args, wfr_lr, corpus)
    out_dir = _EXP09 / "outputs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"parity_pair_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tf_ce = float(payload["transformer"]["best_val_ce"])
    wfr_ce = float(payload["wfr_lm"]["best_val_ce"])
    print(json.dumps(payload["comparison"], indent=2))
    print(json.dumps({"transformer_best_val_ce": tf_ce, "wfr_best_val_ce": wfr_ce}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
