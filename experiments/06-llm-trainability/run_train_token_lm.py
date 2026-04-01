"""
Experiment 06 — LLM Trainability (content-conditioned WFR)
==========================================================

Минимальный «LLM-подобный» тест: next-token на реальном тексте.

Ключевое отличие от Experiment 05:
  - Вводится контент: token -> phase offsets (learned embedding) + позиционные фазы WPE.
  - Это делает модель условной по входным токенам, а не только по позиции.

Целевая (для логов): L = α·CE + β·(1−AvgRC) + γ·EnergyCost (docs/03-theory.md §6).
Оптимизация: по умолчанию только CE (как базовый шаг к LLM-trainability).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
_smoke = _root.parent / "00-smoke-test"
sys.path.insert(0, str(_smoke))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from wfr_core import WFRNetwork  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = _root / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# --- data / model defaults ---
DEFAULT_SEEDS = [42, 43, 44]
SEQ_LEN = 128
BATCH_SIZE = 16
TRAIN_STEPS_PER_EPOCH = 50
VAL_BATCHES = 10
EPOCHS = 60
LR = 3e-3

# Loss weights for reporting (training uses CE by default)
ALPHA = 1.0
BETA = 0.15
GAMMA = 0.1


class ByteTokenizer:
    """Byte-level tokenizer: vocab = 256. Simple, dependency-free, stable."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="ignore")


def load_corpus_text() -> str:
    """
    Lightweight corpus: concatenate a few project docs.
    This keeps the experiment self-contained and reproducible.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "docs" / "03-theory.md",
        Path(__file__).resolve().parents[2] / "docs" / "00-overview.md",
        Path(__file__).resolve().parents[2] / "README.md",
    ]
    chunks: list[str] = []
    for p in candidates:
        try:
            chunks.append(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    text = "\n\n".join(chunks).strip()
    if len(text) < 1000:
        # fallback minimal text
        text = ("WFR Architecture. " * 5000).strip()
    return text


@dataclass
class Batch:
    token_ids: torch.Tensor  # [B, T]


def make_stream_batches(
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    seed: int,
) -> list[Batch]:
    """
    Stream batches from a long token stream: pick random contiguous windows.
    token_ids is 1D on DEVICE.
    """
    assert token_ids.dim() == 1
    n = int(token_ids.numel())
    max_start = n - (seq_len + 1)
    if max_start <= 0:
        raise ValueError("Corpus too small for seq_len")
    g = torch.Generator(device=token_ids.device)
    g.manual_seed(int(seed))
    batches: list[Batch] = []
    for _ in range(num_batches):
        starts = torch.randint(0, max_start, (batch_size,), generator=g, device=token_ids.device)
        windows = []
        for s in starts.tolist():
            windows.append(token_ids[s : s + seq_len].unsqueeze(0))
        x = torch.cat(windows, dim=0)
        batches.append(Batch(token_ids=x))
    return batches


class WFRTokenLM(nn.Module):
    """
    Content-conditioned WFR:
      phases = WPE(positions) + token_phase_offset[token_id]
      logits = head(phases)
    """

    def __init__(self, vocab_size: int, num_phases: int = 32, num_fractal_levels: int = 6, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_phases = num_phases

        self.wfr = WFRNetwork(
            num_phases=num_phases,
            num_fractal_levels=num_fractal_levels,
            num_resonance_layers=num_layers,
            layer_frequencies=PHASE0_FREQ_BALANCED["frequencies"][:num_layers],
            layer_thresholds=PHASE0_FREQ_BALANCED["thresholds"][:num_layers],
            homeostatic_enabled=False,
            spike_rate_target=PHASE0_FREQ_BALANCED["spike_rate_target"],
            homeostatic_eta=PHASE0_FREQ_BALANCED["homeostatic_eta"],
        )
        self.wfr.target_mode = PHASE0_FREQ_BALANCED["target_mode"]

        # Learned content offsets in phase space, initialized small.
        self.token_phase_offset = nn.Embedding(vocab_size, num_phases)
        nn.init.normal_(self.token_phase_offset.weight, mean=0.0, std=0.02)

        self.head = nn.Linear(num_phases, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # positions are 0..T-1 for each batch
        b, t = token_ids.shape
        positions = torch.arange(t, device=token_ids.device).unsqueeze(0).expand(b, t)
        base = self.wfr.encoder(positions)
        content = self.token_phase_offset(token_ids) % (2 * math.pi)
        phases = (base + content) % (2 * math.pi)

        # run resonance stack manually to keep wfr_core untouched
        layer_resonances = []
        layer_spikes = []
        for layer in self.wfr.resonance_layers:
            resonance = layer(phases, target_mode=self.wfr.target_mode)
            spikes = torch.abs(resonance).detach() * 0.0  # placeholder overwritten below
            # Use the same surrogate function as in wfr_core via layer forward already created spikes internally,
            # but it doesn't return it; we approximate energy by threshold crossings from resonance here.
            # This is a pragmatic proxy until wfr_core returns spikes explicitly in resonance layer outputs.
            spikes = (torch.abs(resonance) > layer.spike_threshold).float()
            layer_resonances.append(resonance)
            layer_spikes.append(spikes)

        # RC as in wfr_core
        rc = self.wfr.confidence(phases)
        standing_wave = torch.stack(layer_resonances, dim=0).mean(dim=0)

        out = {
            "phases": phases,
            "layer_resonances": layer_resonances,
            "layer_spikes": layer_spikes,
            "resonance_confidence": rc,
            "standing_wave": standing_wave,
        }
        logits = self.head(phases)
        return logits, out


def task_loss_ce(logits: torch.Tensor, batch_tokens: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        batch_tokens[:, 1:].reshape(-1),
    )


def rc_penalty(out: dict) -> torch.Tensor:
    rc = out["resonance_confidence"]
    return (1.0 - rc) if rc.dim() == 0 else (1.0 - rc).mean()


def energy_cost(out: dict) -> torch.Tensor:
    spikes = out["layer_spikes"]
    rates = torch.stack([s.mean() for s in spikes])
    return rates.mean()


@torch.no_grad()
def evaluate(model: WFRTokenLM, batches: list[Batch]) -> dict:
    model.eval()
    sums = {"ce": 0.0, "rc_term": 0.0, "energy": 0.0}
    for b in batches:
        logits, out = model(b.token_ids)
        ce = task_loss_ce(logits, b.token_ids)
        sums["ce"] += float(ce.item())
        sums["rc_term"] += float(rc_penalty(out).item())
        sums["energy"] += float(energy_cost(out).item())
    n = len(batches)
    ce = sums["ce"] / n
    return {
        "ce": ce,
        "ppl": float(math.exp(min(ce, 20.0))),
        "rc_term": sums["rc_term"] / n,
        "energy": sums["energy"] / n,
        "total_L_report": float(ALPHA * ce + BETA * (sums["rc_term"] / n) + GAMMA * (sums["energy"] / n)),
    }


def plot_curves(history: list[dict], ln_v: float, path_png: Path) -> None:
    epochs = np.arange(1, len(history) + 1)
    val_ce = [h["val"]["ce"] for h in history]
    val_total = [h["val"]["total_L_report"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, val_ce, label="val CE", color="#16a34a")
    ax.plot(epochs, val_total, label="val total L (report)", color="#ea580c", linestyle="--")
    ax.axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Experiment 06 — next-token on real text (byte-level)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def plot_seed_summary(runs: list[dict], ln_v: float, path_png: Path) -> None:
    """Barplot: best val CE per seed, with lnV thresholds."""
    seeds = [r["seed"] for r in runs]
    best = [r["best_val_ce"] for r in runs]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(s) for s in seeds], best, color="#2563eb", alpha=0.9)
    ax.axhline(ln_v, color="gray", linestyle=":", label=f"ln V = {ln_v:.3f}")
    ax.axhline(ln_v - 0.02, color="#ef4444", linestyle="--", label="ln V - 0.02 (strict)")
    ax.set_xlabel("seed")
    ax.set_ylabel("best val CE")
    ax.set_title("Experiment 06 — seed summary (best val CE)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


@torch.no_grad()
def bigram_baseline_ce(token_stream: torch.Tensor, vocab_size: int) -> float:
    """
    Baseline: bigram MLE with add-epsilon smoothing.
    Computes CE over a token stream (1D).
    """
    eps = 1e-3
    counts = torch.zeros((vocab_size, vocab_size), device=token_stream.device, dtype=torch.float32)
    prev = token_stream[:-1]
    nxt = token_stream[1:]
    counts.index_put_((prev, nxt), torch.ones_like(prev, dtype=torch.float32), accumulate=True)
    probs = (counts + eps) / (counts.sum(dim=1, keepdim=True) + eps * vocab_size)
    ce = -torch.log(probs[prev, nxt]).mean()
    return float(ce.item())


def train_one_seed(
    seed: int,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    epochs: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    strict: bool,
) -> dict:
    torch.manual_seed(int(seed))
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    vocab = 256
    ln_v = math.log(vocab)
    model = WFRTokenLM(vocab_size=vocab, num_phases=32, num_fractal_levels=6, num_layers=4).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    val_batches = make_stream_batches(val_ids, seq_len, batch_size, VAL_BATCHES, seed=seed + 99991)

    history: list[dict] = []
    best_val_ce = float("inf")
    best_epoch = 0
    best_snapshot = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_batches = make_stream_batches(train_ids, seq_len, batch_size, TRAIN_STEPS_PER_EPOCH, seed=seed + epoch)
        for b in train_batches:
            opt.zero_grad(set_to_none=True)
            logits, _out = model(b.token_ids)
            loss = task_loss_ce(logits, b.token_ids)  # CE-only optimization
            loss.backward()
            opt.step()

        val = evaluate(model, val_batches)
        train_probe = make_stream_batches(train_ids, seq_len, batch_size, 5, seed=seed + 12345)
        tr = evaluate(model, train_probe)
        history.append({"epoch": epoch, "train_probe": tr, "val": val})

        if val["ce"] < best_val_ce:
            best_val_ce = val["ce"]
            best_epoch = epoch
            best_snapshot = {"val": val, "train_probe": tr}

    pass_strict = best_val_ce < ln_v - 0.02
    pass_soft = best_val_ce < ln_v - 0.001
    passed = pass_strict if strict else pass_soft

    return {
        "seed": int(seed),
        "epochs": int(epochs),
        "best_val_ce": float(best_val_ce),
        "best_epoch": int(best_epoch),
        "pass": bool(passed),
        "criteria": {
            "best_val_ce_below_lnV_minus_0p001": bool(pass_soft),
            "best_val_ce_below_lnV_minus_0p02": bool(pass_strict),
        },
        "best_snapshot": best_snapshot,
        "history": history,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Experiment 06 — LLM trainability (content-conditioned WFR)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--strict", action="store_true", help="Require best val CE < ln(V)-0.02 to PASS")
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (default: 42,43,44). Multiple seeds = stronger evidence.",
    )
    args = p.parse_args()

    tok = ByteTokenizer()
    text = load_corpus_text()
    ids = torch.tensor(tok.encode(text), device=DEVICE, dtype=torch.long)
    # split train/val (contiguous)
    cut = int(ids.numel() * 0.9)
    train_ids = ids[:cut]
    val_ids = ids[cut:]

    vocab = tok.vocab_size
    ln_v = math.log(vocab)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = DEFAULT_SEEDS

    bigram_val_ce = bigram_baseline_ce(val_ids, vocab_size=vocab)

    runs: list[dict] = []
    best_run = None
    for s in seeds:
        r = train_one_seed(
            seed=s,
            train_ids=train_ids,
            val_ids=val_ids,
            epochs=args.epochs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            strict=bool(args.strict),
        )
        runs.append(r)
        if best_run is None or r["best_val_ce"] < best_run["best_val_ce"]:
            best_run = r

        # PNG per seed is always produced after the run finishes.
        png_seed = OUT_DIR / f"llm_trainability_seed{s}_{ts}.png"
        plot_curves(r["history"], ln_v, png_seed)
        r["artifacts"] = {"png_seed_curve": str(png_seed.name)}

    png_summary = OUT_DIR / f"llm_trainability_seed_summary_{ts}.png"
    plot_seed_summary(runs, ln_v, png_summary)

    passed = all(r["pass"] for r in runs)
    result = {
        "mode": "token_lm_byte_multi_seed",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "config": {
            "epochs": args.epochs,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "vocab_size": vocab,
            "strict": bool(args.strict),
            "seeds": seeds,
            "phase0_config": PHASE0_FREQ_BALANCED,
            "note": "content-conditioned: phases = WPE(pos) + token_phase_offset[token]",
        },
        "baselines": {
            "ln_vocab": ln_v,
            "bigram_val_ce": bigram_val_ce,
        },
        "runs": runs,
        "best_run": best_run,
        "pass": bool(passed),
        "artifacts": {
            "json": f"llm_trainability_multi_seed_{ts}.json",
            "png_seed_summary": str(png_summary.name),
        },
    }

    json_path = OUT_DIR / f"llm_trainability_multi_seed_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Experiment 06 — LLM Trainability (multi-seed)")
    print(f"Device: {DEVICE} | vocab=256 | lnV={ln_v:.4f} | epochs={args.epochs} | seeds={seeds}")
    print(f"Baseline bigram val CE: {bigram_val_ce:.4f}")
    print(f"PASS(all seeds): {passed} (strict={args.strict})")
    print(f"JSON: {json_path}")
    print(f"PNG summary: {png_summary}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

