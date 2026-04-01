"""
Прекурсор WFRA-2.0: warm-up (только readout + контентные фазы) + обучение с композитным лоссом.

Протокол данных: как в 06-rfp-protocol-tests — свежие train-батчи каждую эпоху, фиксированный val.

Не реализует полный RFP v1 без backward (это отдельный этап); здесь — честная проверка
**теоретического композитного лосса** через autograd.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(ROOT / "experiments" / "06-rfp-v0"))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from wfr_core import WFRNetwork  # noqa: E402
from wfr_lm import WFRLM  # noqa: E402

from composite_loss import wfra2_composite_loss  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 32
SEQ_LEN = 48
BATCH_SIZE = 16


def _batches(n: int, gen: torch.Generator) -> list[torch.Tensor]:
    return [
        torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=gen, device=DEVICE)
        for _ in range(n)
    ]


@torch.no_grad()
def eval_ce(model: WFRLM, batches: list[torch.Tensor]) -> float:
    model.eval()
    s = 0.0
    k = 0
    for batch in batches:
        st = model(batch, batch)
        ce = F.cross_entropy(
            st.logits[:, :-1].reshape(-1, st.logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )
        s += float(ce.item())
        k += 1
    return s / max(k, 1)


def total_grad_norm(model: torch.nn.Module) -> float:
    t = 0.0
    for p in model.parameters():
        if p.grad is not None:
            t += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(t) if t > 0 else 0.0


def set_core_frozen(model: WFRLM, frozen: bool) -> None:
    for p in model.core.parameters():
        p.requires_grad = not frozen


def set_head_and_content_trainable(model: WFRLM) -> None:
    for p in model.head.parameters():
        p.requires_grad = True
    for p in model.token_phase_offset.parameters():
        p.requires_grad = True


@dataclass
class RunRecord:
    mode: str
    best_val_ce: float
    final_val_ce: float
    max_grad: float
    warmup_epochs: int
    epochs: int
    loss_parts_last: Optional[dict[str, float]] = None


def train_one_mode(
    mode: str,
    epochs: int,
    warmup_epochs: int,
    train_seed: int,
    val_seed: int,
    lr: float,
    lr_warmup: float,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> RunRecord:
    torch.manual_seed(train_seed)
    val_gen = torch.Generator(device=DEVICE)
    val_gen.manual_seed(val_seed)
    val_batches = _batches(4, val_gen)

    cfg = PHASE0_FREQ_BALANCED
    core = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=4,
        layer_frequencies=cfg["frequencies"][:4],
        layer_thresholds=cfg["thresholds"][:4],
        homeostatic_enabled=True,
        spike_rate_target=0.25,
        homeostatic_eta=cfg["homeostatic_eta"],
        homeostatic_always_on=True,
    )
    core.target_mode = cfg["target_mode"]
    model = WFRLM(core, vocab_size=VOCAB_SIZE, num_phases=16).to(DEVICE)

    best_val = float("inf")
    max_g = 0.0
    last_parts: Optional[dict[str, float]] = None

    # ---------- Warm-up ----------
    if warmup_epochs > 0:
        set_core_frozen(model, True)
        set_head_and_content_trainable(model)
        opt = torch.optim.AdamW(
            list(model.head.parameters()) + list(model.token_phase_offset.parameters()),
            lr=lr_warmup,
        )
        for ep in range(warmup_epochs):
            model.train()
            g = torch.Generator(device=DEVICE)
            g.manual_seed(train_seed + ep * 100_003 + 17)
            train_batches = _batches(8, g)
            for batch in train_batches:
                opt.zero_grad(set_to_none=True)
                state = model(batch, batch)
                ce = F.cross_entropy(
                    state.logits[:, :-1].reshape(-1, state.logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
                ce.backward()
                max_g = max(max_g, total_grad_norm(model))
                opt.step()

    # ---------- Main ----------
    set_core_frozen(model, False)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        g = torch.Generator(device=DEVICE)
        g.manual_seed(train_seed + ep * 100_003)
        train_batches = _batches(8, g)
        for batch in train_batches:
            opt.zero_grad(set_to_none=True)
            state = model(batch, batch)
            mean_spk = state.spikes.mean()

            if mode == "ce_only":
                loss = F.cross_entropy(
                    state.logits[:, :-1].reshape(-1, state.logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            else:
                loss, parts = wfra2_composite_loss(
                    state.logits,
                    batch,
                    state.phases,
                    mean_spk,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                last_parts = {k: float(v) for k, v in parts.items() if k in ("l_task", "l_res", "l_energy", "l_phase")}

            loss.backward()
            max_g = max(max_g, total_grad_norm(model))
            opt.step()

        vce = eval_ce(model, val_batches)
        best_val = min(best_val, vce)

    final_v = eval_ce(model, val_batches)
    return RunRecord(
        mode=mode,
        best_val_ce=best_val,
        final_val_ce=final_v,
        max_grad=max_g,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        loss_parts_last=last_parts,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="WFRA-2.0 precursor — composite loss + warm-up")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--warmup-epochs", type=int, default=8)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=4242)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--lr-warmup", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    out_dir = _EXP / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    common_kw = dict(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        lr=args.lr,
        lr_warmup=args.lr_warmup,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    r_ce = train_one_mode("ce_only", **common_kw)
    r_wf = train_one_mode("wfra2_composite", **common_kw)

    h = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report: dict[str, Any] = {
        "experiment": "07-wfra2-precursor",
        "note": "Сравнение CE-only vs композитный лосс WFRA-2 (без полного RFP v1 по слоям).",
        "hyper": h,
        "baseline_ce_only": asdict(r_ce),
        "wfra2_composite": asdict(r_wf),
        "delta_final_val_ce": r_ce.final_val_ce - r_wf.final_val_ce,
    }

    path = args.out_json or (out_dir / f"wfra2_precursor_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
