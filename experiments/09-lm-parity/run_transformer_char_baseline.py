"""
Exp 09 — минимальный causal Transformer (char LM) на WikiText-2 с тем же протоколом окон, что WFRLM.

Дорожная карта: docs/12-wfr-llm-breakthrough-roadmap.md (этап A).

Запуск из корня: python experiments/09-lm-parity/run_transformer_char_baseline.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
_EXP08 = ROOT / "experiments" / "08-wikitext-rfp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_EXP08))

from wikitext_loader import WikiTextCharCorpus  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SEED = 4242
TRAIN_SEED_BASE = 42


@dataclass
class BaselineRun:
    best_val_ce: float
    final_val_ce: float
    epochs: int
    d_model: int
    nhead: int
    nlayers: int
    dim_feedforward: int
    num_params: int
    train_seed: int
    val_seed: int
    init_seed: int


class TinyCausalTransformer(nn.Module):
    """Batch-first TransformerEncoder stack с is_causal=True (PyTorch 2.x)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_feedforward: int,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        h = self.tok_emb(x) + self.pos_emb(pos)
        # Causal mask: совместимо с разными версиями PyTorch (см. nn.Transformer docs).
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.encoder(h, mask=causal, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


@torch.no_grad()
def eval_ce(model: nn.Module, batches: list[torch.Tensor]) -> float:
    model.eval()
    s = 0.0
    n = 0
    for batch in batches:
        logits = model(batch)
        ce = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )
        s += float(ce.item())
        n += 1
    return s / max(1, n)


def train_run(
    corpus: WikiTextCharCorpus,
    *,
    epochs: int,
    seq_len: int,
    batch_size: int,
    num_train_batches: int,
    num_val_batches: int,
    lr: float,
    d_model: int,
    nhead: int,
    nlayers: int,
    dim_feedforward: int,
    init_seed: int,
    train_seed: int,
    val_seed: int,
) -> tuple[BaselineRun, list[dict]]:
    torch.manual_seed(init_seed)
    v = corpus.vocab_size
    model = TinyCausalTransformer(
        v, d_model=d_model, nhead=nhead, nlayers=nlayers, dim_feedforward=dim_feedforward, max_len=seq_len + 32
    ).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    val_batches = corpus.make_val_batches(num_val_batches, batch_size, seq_len, val_seed, DEVICE)

    best_val = float("inf")
    log_rows: list[dict] = []

    for ep in range(epochs):
        model.train()
        train_batches = corpus.make_train_batches_for_epoch(
            num_train_batches, batch_size, seq_len, ep, train_seed, DEVICE
        )
        for batch in train_batches:
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_ce = eval_ce(model, val_batches)
        best_val = min(best_val, val_ce)
        log_rows.append({"epoch": ep + 1, "val_ce": val_ce, "best_so_far": best_val})

    final_val = float(log_rows[-1]["val_ce"]) if log_rows else float("inf")
    br = BaselineRun(
        best_val_ce=float(best_val),
        final_val_ce=final_val,
        epochs=epochs,
        d_model=d_model,
        nhead=nhead,
        nlayers=nlayers,
        dim_feedforward=dim_feedforward,
        num_params=n_param,
        train_seed=train_seed,
        val_seed=val_seed,
        init_seed=init_seed,
    )
    return br, log_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-train-batches", type=int, default=6)
    ap.add_argument("--num-val-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--dim-ff", type=int, default=512)
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED_BASE)
    ap.add_argument("--val-seed", type=int, default=VAL_SEED)
    args = ap.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 6)
        args.num_train_batches = min(args.num_train_batches, 4)
        args.num_val_batches = min(args.num_val_batches, 2)
        args.batch_size = min(args.batch_size, 8)

    if args.d_model % args.nhead != 0:
        raise SystemExit("d_model must be divisible by nhead")

    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=256)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    br, log_rows = train_run(
        corpus,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_train_batches=args.num_train_batches,
        num_val_batches=args.num_val_batches,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dim_feedforward=args.dim_ff,
        init_seed=args.init_seed,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "experiment": "09-lm-parity",
        "architecture": "TinyCausalTransformer",
        "protocol_note": "Same WikiTextCharCorpus window sampling as run_wikitext_train (fresh train / fixed val).",
        "corpus_meta": corpus.meta,
        "device": str(DEVICE),
        "ln_vocab": math.log(corpus.vocab_size),
        "metrics": asdict(br),
        "epoch_log": log_rows,
    }
    path = out_dir / f"transformer_baseline_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
