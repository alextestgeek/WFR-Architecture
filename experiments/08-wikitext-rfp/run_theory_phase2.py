"""
Exp08 — Phase 2: теория ↔ измерения (один воспроизводимый suite).

Порядок прогонов (развязка факторов):
  1) Adam-only vs Adam+RFP на эталонной схеме `absolute / normal`.
  2) Head-to-head `absolute` vs `token_as_pos` при включённом RFP.
  3) Контроль деградации: `absolute / off` (нижняя граница по контенту).

`--minutes` — суммарный бюджет wall-time ориентира; делится на число прогонов → эпохи/seq/батчи
(как в `test_input_schemes.py`, без комбинаторного взрыва).

По умолчанию фаза **ab_rfp** дублируется: обучение с полным \(L\) (CE+RC+energy) и **train CE-only**
(тот же Adam/RFP/данные; val CE всегда чистый CE). Сравнение в `paired_ab_rfp_objectives` и в секциях
`ab_rfp_full_L` / `ab_rfp_ce_only_train`. Отключить twin: `--no-ce-only-twin`.

Запуск:
  python experiments/08-wikitext-rfp/run_theory_phase2.py --quick
  python experiments/08-wikitext-rfp/run_theory_phase2.py --minutes 90 --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
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
from test_input_schemes import _artifact_stem, plot_epoch_dynamics  # noqa: E402
from wikitext_loader import WikiTextCharCorpus  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    phase: str
    label: str
    pos_mode: str
    content_mode: str
    content_scale: float
    use_rfp: bool
    loss_mode: str = "full"  # "full" | "ce_only"


def _apply_minutes_budget(
    *,
    minutes: int,
    n_runs: int,
    quick: bool,
    base_epochs: int,
    base_n_train: int,
    base_n_val: int,
    base_seq: int,
) -> tuple[int, int, int, int]:
    epochs = base_epochs
    n_train = base_n_train
    n_val = base_n_val
    seq_len = base_seq
    if quick or minutes <= 0 or n_runs <= 0:
        return epochs, n_train, n_val, seq_len
    mpr = minutes / n_runs
    if mpr >= 4:
        epochs = max(epochs, 24)
        n_train = max(n_train, 14)
        n_val = max(n_val, 6)
        seq_len = max(seq_len, 80)
    if mpr >= 6:
        epochs = max(epochs, 36)
        n_train = max(n_train, 16)
        n_val = max(n_val, 8)
        seq_len = max(seq_len, 96)
    if mpr >= 9:
        epochs = max(epochs, 48)
        n_train = max(n_train, 18)
        n_val = max(n_val, 8)
        seq_len = max(seq_len, 112)
    if mpr >= 12:
        epochs = max(epochs, 72)
        n_train = max(n_train, 20)
        n_val = max(n_val, 8)
        seq_len = max(seq_len, 128)
    if mpr >= 18:
        epochs = max(epochs, 96)
        n_train = max(n_train, 24)
        n_val = max(n_val, 10)
        seq_len = max(seq_len, 128)
    return epochs, n_train, n_val, seq_len


def _row_at_min_val_ce(epoch_log: list[dict]) -> dict:
    if not epoch_log:
        return {}
    best = min(epoch_log, key=lambda e: (float(e["val_ce"]), int(e["epoch"])))
    last = epoch_log[-1]
    return {
        "epoch_min_val_ce": int(best["epoch"]),
        "val_ce_at_best": float(best["val_ce"]),
        "val_rc_at_best": float(best["val_rc"]),
        "spike_at_best": float(best["spike_rate"]),
        "train_ce_at_best": float(best["train_ce_mean"]),
        "final_val_ce": float(last["val_ce"]),
        "final_val_rc": float(last["val_rc"]),
        "final_spike": float(last["spike_rate"]),
    }


def _wave_health(
    spike: float,
    val_rc: float,
    *,
    spike_target: float,
    spike_lo: float = 0.04,
    spike_hi: float = 0.55,
    rc_lo: float = 0.15,
) -> dict[str, object]:
    """Грубая проверка: лучше по CE не за счёт «мёртвого» спайка/RC."""
    in_spike_band = spike_lo <= spike <= spike_hi
    rc_ok = val_rc >= rc_lo
    return {
        "spike_target": spike_target,
        "spike_in_band": in_spike_band,
        "rc_not_collapsed": rc_ok,
        "healthy_guess": bool(in_spike_band and rc_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--minutes", type=int, default=0, help="суммарный ориентир времени на весь suite")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--seeds-off", type=str, default="42,43", help="сиды только для фазы content_off")
    ap.add_argument("--rfp-version", type=str, default="v03", choices=("v02", "v03"))
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--spike-target", type=float, default=0.25)
    ap.add_argument(
        "--no-ce-only-twin",
        action="store_true",
        help="не дублировать фазу ab_rfp прогоном с train CE-only (по умолчанию twin включён для честного сравнения цели обучения)",
    )
    args = ap.parse_args()
    ce_only_twin = not args.no_ce_only_twin

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    seeds_off = [int(s.strip()) for s in args.seeds_off.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No seeds in --seeds")
    if not seeds_off:
        raise SystemExit("No seeds in --seeds-off")

    def _ab_adam_label(lm: str) -> str:
        base = f"Adam only | absolute/normal ({args.rfp_version} stack)"
        return base + (" | train CE-only" if lm == "ce_only" else " | full L train")

    def _ab_rfp_label(lm: str) -> str:
        base = f"Adam+RFP {args.rfp_version} | absolute/normal"
        return base + (" | train CE-only" if lm == "ce_only" else " | full L train")

    ab_loss_modes: tuple[str, ...] = ("full", "ce_only") if ce_only_twin else ("full",)

    runs: list[tuple[RunSpec, int]] = []
    # 1) RFP vs baseline (+ опционально тот же протокол с train CE-only)
    for s in seeds:
        for lm in ab_loss_modes:
            runs.append(
                (
                    RunSpec(
                        "ab_rfp",
                        _ab_adam_label(lm),
                        "absolute",
                        "normal",
                        1.0,
                        False,
                        loss_mode=lm,
                    ),
                    s,
                )
            )
            runs.append(
                (
                    RunSpec(
                        "ab_rfp",
                        _ab_rfp_label(lm),
                        "absolute",
                        "normal",
                        1.0,
                        True,
                        loss_mode=lm,
                    ),
                    s,
                )
            )
    # 2) absolute vs token_as_pos — absolute/normal+RFP уже в фазе A; здесь только token_as_pos.
    for s in seeds:
        runs.append(
            (
                RunSpec(
                    "h2h_pos",
                    f"token_as_pos/normal + RFP {args.rfp_version}",
                    "token_as_pos",
                    "normal",
                    1.0,
                    True,
                    loss_mode="full",
                ),
                s,
            )
        )
    # 3) content off
    for s in seeds_off:
        runs.append(
            (
                RunSpec(
                    "control_off",
                    f"absolute/off + RFP {args.rfp_version} (control)",
                    "absolute",
                    "off",
                    0.0,
                    True,
                    loss_mode="full",
                ),
                s,
            )
        )

    n_runs = len(runs)
    base_e = 6 if args.quick else args.epochs
    base_nt = 4 if args.quick else 10
    base_nv = 2 if args.quick else 6
    base_seq = 48 if args.quick else 64
    epochs, n_train, n_val, seq_len = _apply_minutes_budget(
        minutes=args.minutes,
        n_runs=n_runs,
        quick=args.quick,
        base_epochs=base_e,
        base_n_train=base_nt,
        base_n_val=base_nv,
        base_seq=base_seq,
    )

    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=256)
    out_dir = _EXP / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.save_vocab_json(out_dir / "wikitext08_vocab_meta.json")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_png = not args.no_png
    per_run: list[dict[str, object]] = []

    for spec, seed in runs:
        # Фаза + adam/rfp + цель обучения — уникальные артефакты при twin full/ce_only.
        rfp_tag = "rfp" if spec.use_rfp else "adam"
        obj_tag = "ceonly" if spec.loss_mode == "ce_only" else "fullL"
        stem = f"phase2_{spec.phase}_{rfp_tag}_{obj_tag}_" + _artifact_stem(spec.pos_mode, spec.content_mode, spec.content_scale, seed, ts)
        dyn_png = out_dir / f"{stem}_dynamics.png"
        label = spec.label
        m = train_wikitext_run(
            corpus,
            epochs=epochs,
            seq_len=seq_len,
            batch_size=12 if not args.quick else 8,
            num_train_batches=n_train,
            num_val_batches=n_val,
            use_rfp=spec.use_rfp,
            rfp_interval=8,
            rfp_version=args.rfp_version,
            online_rfp=False,
            homeostatic_always_on=True,
            spike_rate_target=args.spike_target,
            init_seed=seed,
            train_seed=42,
            val_seed=4242,
            lr=0.02,
            save_png=save_png,
            plot_path=out_dir / f"{stem}_curves.png" if save_png else None,
            epoch_json_path=out_dir / f"{stem}.json",
            verbose_epochs=bool(args.minutes >= 60 and not args.quick),
            integrity_checks=True,
            manifest_path=out_dir / f"{stem}_manifest.json",
            epoch_csv_path=out_dir / f"{stem}.csv",
            pos_mode=spec.pos_mode,
            content_mode=spec.content_mode,
            content_scale=spec.content_scale,
            loss_mode=spec.loss_mode,
        )
        elog = m.epoch_log
        if elog:
            plot_epoch_dynamics(
                elog,
                dyn_png,
                title=f"{spec.phase}: {label} | seed={seed} | {epochs} ep",
                vocab_size=corpus.vocab_size,
            )
        row_best = _row_at_min_val_ce(elog)
        health_best = _wave_health(
            row_best.get("spike_at_best", float("nan")),
            row_best.get("val_rc_at_best", float("nan")),
            spike_target=args.spike_target,
        )
        health_final = _wave_health(
            row_best.get("final_spike", float("nan")),
            row_best.get("final_val_rc", float("nan")),
            spike_target=args.spike_target,
        )
        per_run.append(
            {
                "phase": spec.phase,
                "label": label,
                "seed": seed,
                "use_rfp": spec.use_rfp,
                "loss_mode": spec.loss_mode,
                "best_val_ce": float(m.metrics.best_val_ce),
                "metrics_at_min_val_ce": row_best,
                "wave_health_at_best": health_best,
                "wave_health_final": health_final,
                "dynamics_png": dyn_png.name if dyn_png.is_file() else None,
                "epoch_csv": f"{stem}.csv",
            }
        )

    def summarize_phase(phase: str, *, loss_mode: str | None = None) -> list[dict[str, object]]:
        subset = [r for r in per_run if r["phase"] == phase]
        if loss_mode is not None:
            subset = [r for r in subset if str(r.get("loss_mode", "full")) == loss_mode]
        by_label: dict[str, list[float]] = {}
        for r in subset:
            lb = str(r["label"])
            by_label.setdefault(lb, []).append(float(r["best_val_ce"]))
        rows = []
        for lb, ces in sorted(by_label.items()):
            rows.append(
                {
                    "mode": lb,
                    "best_val_ce_mean": statistics.mean(ces),
                    "best_val_ce_std": statistics.pstdev(ces) if len(ces) > 1 else 0.0,
                    "n": len(ces),
                }
            )
        return rows

    def summarize_h2h() -> list[dict[str, object]]:
        abs_label = f"absolute/normal + RFP {args.rfp_version} (from phase ab_rfp, full L)"
        abs_ces = [
            float(r["best_val_ce"])
            for r in per_run
            if r["phase"] == "ab_rfp"
            and str(r["label"]).startswith("Adam+RFP")
            and str(r.get("loss_mode", "full")) == "full"
        ]
        tok = [r for r in per_run if r["phase"] == "h2h_pos"]
        tok_ces = [float(r["best_val_ce"]) for r in tok]
        rows: list[dict[str, object]] = []
        if abs_ces:
            rows.append(
                {
                    "mode": abs_label,
                    "best_val_ce_mean": statistics.mean(abs_ces),
                    "best_val_ce_std": statistics.pstdev(abs_ces) if len(abs_ces) > 1 else 0.0,
                    "n": len(abs_ces),
                }
            )
        if tok_ces:
            lb = str(tok[0]["label"]) if tok else f"token_as_pos/normal + RFP {args.rfp_version}"
            rows.append(
                {
                    "mode": lb,
                    "best_val_ce_mean": statistics.mean(tok_ces),
                    "best_val_ce_std": statistics.pstdev(tok_ces) if len(tok_ces) > 1 else 0.0,
                    "n": len(tok_ces),
                }
            )
        return rows

    def paired_ab_full_vs_ce() -> list[dict[str, object]]:
        if not ce_only_twin:
            return []
        ab = [r for r in per_run if r["phase"] == "ab_rfp"]
        out: list[dict[str, object]] = []
        for s in seeds:
            for urfp in (False, True):
                full = next(
                    (x for x in ab if int(x["seed"]) == s and bool(x["use_rfp"]) == urfp and x["loss_mode"] == "full"),
                    None,
                )
                ce_o = next(
                    (
                        x
                        for x in ab
                        if int(x["seed"]) == s and bool(x["use_rfp"]) == urfp and x["loss_mode"] == "ce_only"
                    ),
                    None,
                )
                if not full or not ce_o:
                    continue
                mfull = full["metrics_at_min_val_ce"]
                mce = ce_o["metrics_at_min_val_ce"]
                out.append(
                    {
                        "seed": s,
                        "use_rfp": urfp,
                        "best_val_ce_full_L": float(full["best_val_ce"]),
                        "best_val_ce_ce_only_train": float(ce_o["best_val_ce"]),
                        "delta_ce_best_full_minus_ce_only": float(full["best_val_ce"]) - float(ce_o["best_val_ce"]),
                        "val_rc_at_best_epoch_full_L": mfull.get("val_rc_at_best"),
                        "val_rc_at_best_epoch_ce_only_train": mce.get("val_rc_at_best"),
                    }
                )
        return out

    sections: dict[str, object] = {
        "ab_rfp_full_L": summarize_phase("ab_rfp", loss_mode="full"),
        "h2h_pos": summarize_h2h(),
        "control_off": summarize_phase("control_off"),
    }
    if ce_only_twin:
        sections["ab_rfp_ce_only_train"] = summarize_phase("ab_rfp", loss_mode="ce_only")
    sections["ab_rfp"] = sections["ab_rfp_full_L"]

    bar_paths: dict[str, str | None] = {}
    if save_png:
        for phase, title in (
            ("ab_rfp_full_L", "Phase2 A: Adam vs RFP (absolute/normal, full L)"),
            ("ab_rfp_ce_only_train", "Phase2 A: Adam vs RFP (absolute/normal, train CE-only)"),
            ("h2h_pos", "Phase2 B: absolute vs token_as_pos (RFP)"),
            ("control_off", "Phase2 C: content off (control)"),
        ):
            if phase == "ab_rfp_ce_only_train" and not ce_only_twin:
                continue
            rows = sections.get(phase) or []
            plot_rows = [{"mode": r["mode"], "best_val_ce": float(r["best_val_ce_mean"])} for r in rows]
            if plot_rows:
                p = out_dir / f"wikitext08_phase2_bar_{phase}_{ts}.png"
                plot_ab_modes_bar(plot_rows, p, title=f"{title} ({epochs} ep)", vocab_size=corpus.vocab_size)
                bar_paths[phase] = p.name
            else:
                bar_paths[phase] = None
    if bar_paths.get("ab_rfp_full_L"):
        bar_paths["ab_rfp"] = bar_paths["ab_rfp_full_L"]

    paired = paired_ab_full_vs_ce()
    report = {
        "protocol": "Exp08 theory phase2",
        "order": ["ab_rfp", "h2h_pos", "control_off"],
        "ce_only_twin_enabled": ce_only_twin,
        "paired_ab_rfp_objectives": paired,
        "objective_comparison_note": (
            "delta_ce_best_full_minus_ce_only: при отрицательном значении полное L даёт лучший val CE, чем train CE-only "
            "(при прочих равных); val RC в строке paired сравнивает диагностику на эпохе лучшего val CE для каждой цели."
            if ce_only_twin
            else "Twin CE-only отключён (--no-ce-only-twin)."
        ),
        "hypotheses": {
            "H1": "RFP улучшает или стабилизирует val CE относительно Adam-only при той же входной схеме.",
            "H1b": "При включённом twin: вклад полного L vs чистого CE в val CE и в RC на лучшей эпохе виден по paired_ab_rfp_objectives.",
            "H2": "Корректная разметка позиций (absolute) не хуже token_as_pos при достаточной длине обучения.",
            "H3": "Отключение content (off) даёт контролируемую деградацию; улучшение CE без катастрофы RC/spike.",
        },
        "success_criteria": {
            "numerical": "Нет NaN; best_val_ce заметно ниже ln(V); различия между режимами воспроизводимы по нескольким сидам.",
            "wave_health": "На эпохе минимума val CE spike и RC не выглядят тривиально «мёртвыми» (см. wave_health_*). При full L val RC частично под целевой функцией; при train CE-only — только диагностика.",
        },
        "epochs": epochs,
        "seq_len": seq_len,
        "num_train_batches": n_train,
        "num_val_batches": n_val,
        "minutes_total": args.minutes,
        "n_runs": n_runs,
        "seeds_main": seeds,
        "seeds_control_off": seeds_off,
        "sections": sections,
        "per_run": per_run,
        "bar_png": bar_paths,
        "artifacts_note": "phase2_*dynamics.png — CE/acc/train/RC/spike; phase2_bar_* — сравнение средних best_val_ce по меткам.",
        "dedup_note": "Фаза h2h_pos не дублирует прогон absolute/normal+RFP: метрики absolute (full L) берутся из ab_rfp_full_L. Секция sections['ab_rfp'] = ab_rfp_full_L для обратной совместимости.",
    }
    out_json = out_dir / f"wikitext08_phase2_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
