"""
Experiment 06 — RFP v0: Adam + composite loss + периодический RFP-шаг.

Запуск из корня репозитория или с добавлением ROOT в PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from wfr_core import WFRNetwork  # noqa: E402
from wfr_lm import WFRLM  # noqa: E402
from wfr_rfp import apply_rfp_deltas, rfp_step, rfp_step_v01  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = 32
SEQ_LEN = 48
BATCH_SIZE = 16
SEED = 42
ALPHA = 1.0
BETA = 0.15
GAMMA = 0.1


@dataclass
class RunMetrics:
    best_val_ce: float
    final_val_rc: float
    mean_spike_rate: float
    max_grad_l2: float
    epochs: int
    mode: str


def total_grad_l2_norm(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(s) if s > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: WFRLM,
    batches: list[torch.Tensor],
) -> dict:
    model.eval()
    ce_sum = 0.0
    rc_sum = 0.0
    sp_sum = 0.0
    n = 0
    for batch in batches:
        state = model(batch, batch)
        ce = F.cross_entropy(
            state.logits[:, :-1].reshape(-1, state.logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )
        rc = state.rc
        if rc.dim() > 0:
            rc = rc.mean()
        sp = state.spikes.mean()
        ce_sum += float(ce.item())
        rc_sum += float(rc.item())
        sp_sum += float(sp.item())
        n += 1
    return {
        "val_ce": ce_sum / n,
        "val_rc": rc_sum / n,
        "spike_rate": sp_sum / n,
    }


def train_run(
    epochs: int,
    use_rfp: bool,
    rfp_interval: int,
    rfp_v01: bool,
    online_rfp: bool,
    homeostatic_always_on: bool,
    spike_rate_target: float,
    lr: float = 0.05,
    seed: int = SEED,
) -> RunMetrics:
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)

    cfg = PHASE0_FREQ_BALANCED
    core = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=4,
        layer_frequencies=cfg["frequencies"][:4],
        layer_thresholds=cfg["thresholds"][:4],
        homeostatic_enabled=True,
        spike_rate_target=spike_rate_target,
        homeostatic_eta=cfg["homeostatic_eta"],
        homeostatic_always_on=homeostatic_always_on,
    )
    core.target_mode = cfg["target_mode"]

    model = WFRLM(core, vocab_size=VOCAB_SIZE, num_phases=16).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def make_batches(num: int) -> list[torch.Tensor]:
        return [
            torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=gen, device=DEVICE)
            for _ in range(num)
        ]

    train_batches = make_batches(8)
    val_batches = make_batches(3)

    max_g = 0.0
    best_val_ce = float("inf")
    final_rc = 0.0
    mean_sp = 0.0
    global_step = 0

    mode = "baseline"
    if use_rfp:
        mode = "rfp_online" if online_rfp else f"rfp_every_{rfp_interval}"
        if rfp_v01:
            mode += "_v01"

    for _ep in range(epochs):
        model.train()
        for batch in train_batches:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            state = model(batch, batch)
            ce_loss = F.cross_entropy(
                state.logits[:, :-1].reshape(-1, state.logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            rc_term = 1.0 - state.rc if state.rc.dim() == 0 else (1.0 - state.rc).mean()
            total = ALPHA * ce_loss + BETA * rc_term + GAMMA * state.energy
            total.backward()
            g2 = total_grad_l2_norm(model)
            max_g = max(max_g, g2)
            opt.step()

            do_rfp = use_rfp and (
                online_rfp or (global_step % rfp_interval == 0)
            )
            if do_rfp:
                ce_t = ce_loss.detach()
                if rfp_v01:
                    deltas = rfp_step_v01(
                        model.core,
                        state,
                        ce_t,
                        layer_resonance_means=state.layer_resonance_means,
                    )
                else:
                    deltas = rfp_step(model.core, state, ce_t)
                apply_rfp_deltas(model.core, deltas)

        ev = evaluate(model, val_batches)
        best_val_ce = min(best_val_ce, ev["val_ce"])
        final_rc = ev["val_rc"]
        mean_sp = ev["spike_rate"]

    return RunMetrics(
        best_val_ce=best_val_ce,
        final_val_rc=final_rc,
        mean_spike_rate=mean_sp,
        max_grad_l2=max_g,
        epochs=epochs,
        mode=mode,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="RFP v0 training (Experiment 06)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--no-rfp", action="store_true", help="Adam only (baseline)")
    p.add_argument("--rfp-interval", type=int, default=8)
    p.add_argument("--online-rfp", action="store_true")
    p.add_argument("--rfp-v01", action="store_true", help="cos(Δ) term on phase_bias")
    p.add_argument(
        "--no-homeostatic-always-on",
        action="store_true",
        help="отключить homeostatic в train (по умолчанию включён для RFP v0)",
    )
    p.add_argument("--spike-rate-target", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    m = train_run(
        epochs=args.epochs,
        use_rfp=not args.no_rfp,
        rfp_interval=args.rfp_interval,
        rfp_v01=args.rfp_v01,
        online_rfp=args.online_rfp,
        homeostatic_always_on=not args.no_homeostatic_always_on,
        spike_rate_target=args.spike_rate_target,
        lr=args.lr,
        seed=args.seed,
    )
    d = asdict(m)
    print(json.dumps(d, indent=2))
    out = args.out_json or (Path(__file__).parent / "outputs" / "last_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


if __name__ == "__main__":
    main()
