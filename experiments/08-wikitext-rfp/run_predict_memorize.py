"""
Experiment 08 (addon) — измеримый тест «предсказывает / запоминает / обобщает».

Цели:
- Показать **явно** next-token prediction: CE + accuracy@1 на фиксированном val/test.
- Показать **memorization**: перетрен на маленьком фиксированном train-holdout (accuracy→~1).
- Показать отсутствие «подкрутки»: фиксируем SHA256 батчей val/test/mem и сиды в манифесте.

Запуск (локально или на GPU):
  python experiments/08-wikitext-rfp/run_predict_memorize.py --quick
  python experiments/08-wikitext-rfp/run_predict_memorize.py --minutes 10
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "00-smoke-test"))
sys.path.insert(0, str(ROOT / "experiments" / "06-rfp-v0"))

from phase0_best_config import PHASE0_FREQ_BALANCED  # noqa: E402
from run_rfp_training import (  # noqa: E402
    ALPHA,
    BETA,
    DEVICE,
    GAMMA,
    total_grad_l2_norm,
)
from wfr.core import WFRNetwork  # noqa: E402
from wfr_lm import WFRLM  # noqa: E402
from wfr_rfp import apply_rfp_deltas, rfp_step_v03  # noqa: E402

from wikitext_integrity import tensor_batches_fingerprint  # noqa: E402
from wikitext_loader import WikiTextCharCorpus  # noqa: E402


def make_positions(tokens: torch.Tensor, *, pos_mode: str) -> torch.Tensor:
    if tokens.dim() != 2:
        raise ValueError("tokens must be [B,T]")
    b, t = tokens.shape
    if pos_mode == "absolute":
        return torch.arange(t, device=tokens.device, dtype=tokens.dtype).unsqueeze(0).expand(b, -1)
    if pos_mode == "token_as_pos":
        return tokens
    raise ValueError(f"Unknown pos_mode: {pos_mode}")


@dataclass
class EvalMetrics:
    ce: float
    acc1: float
    rc: float
    energy: float
    spike_rate: float


@torch.no_grad()
def eval_batches(model: WFRLM, batches: list[torch.Tensor]) -> EvalMetrics:
    model.eval()
    ce_sum = acc_sum = rc_sum = e_sum = sp_sum = 0.0
    n_tok = 0
    n_batches = 0
    for x in batches:
        pos = make_positions(x, pos_mode="absolute")
        st = model(pos, x)
        logits = st.logits[:, :-1]
        tgt = x[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        pred = logits.argmax(dim=-1)
        acc = (pred == tgt).float().mean()
        rc = st.rc if st.rc.dim() == 0 else st.rc.mean()
        e = st.energy if st.energy.dim() == 0 else st.energy.mean()
        sp = st.spikes.mean()
        ce_sum += float(ce.item())
        acc_sum += float(acc.item())
        rc_sum += float(rc.item())
        e_sum += float(e.item())
        sp_sum += float(sp.item())
        n_batches += 1
        n_tok += int(tgt.numel())
    inv = 1.0 / max(1, n_batches)
    return EvalMetrics(
        ce=ce_sum * inv,
        acc1=acc_sum * inv,
        rc=rc_sum * inv,
        energy=e_sum * inv,
        spike_rate=sp_sum * inv,
    )


@torch.no_grad()
def _sample_next_token(
    logits_1d: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
) -> int:
    """Top-k sampling для предотвращения вырождения greedy, без изменения модели."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = logits_1d / float(temperature)
    if top_k and top_k > 0:
        k = min(int(top_k), x.numel())
        v, idx = torch.topk(x, k)
        probs = torch.softmax(v, dim=-1)
        pick = int(torch.multinomial(probs, 1).item())
        return int(idx[pick].item())
    probs = torch.softmax(x, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(
    model: WFRLM,
    corpus: WikiTextCharCorpus,
    *,
    prefix: str,
    gen_len: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    device: torch.device = DEVICE,
) -> str:
    ids = corpus.vocab.encode_slice(prefix)
    # генерация: каждый шаг предсказываем следующий символ по волновому состоянию
    for _ in range(gen_len):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        pos = make_positions(x, pos_mode="absolute")
        st = model(pos, x)
        nxt = _sample_next_token(st.logits[0, -1], temperature=temperature, top_k=top_k)
        ids.append(nxt)
    # декод
    out = []
    for i in ids:
        if i < 0 or i >= len(corpus.vocab.id2char):
            out.append("?")
        else:
            ch = corpus.vocab.id2char[i]
            out.append("�" if ch in ("<pad>", "<unk>") else ch)
    return "".join(out)


def plot_curves(history: list[dict[str, float]], out_png: Path, vocab_size: int) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in history]
    ln_v = math.log(vocab_size)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(ep, [h["mem_ce"] for h in history], label="mem CE", color="#7c3aed")
    ax.plot(ep, [h["val_ce"] for h in history], label="val CE", color="#16a34a")
    ax.plot(ep, [h["test_ce"] for h in history], label="test CE", color="#2563eb")
    ax.axhline(ln_v, color="gray", linestyle=":", label=f"ln(V)={ln_v:.3f}")
    ax.set_title("Cross-entropy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(ep, [h["mem_acc1"] for h in history], label="mem acc@1", color="#7c3aed")
    ax.plot(ep, [h["val_acc1"] for h in history], label="val acc@1", color="#16a34a")
    ax.plot(ep, [h["test_acc1"] for h in history], label="test acc@1", color="#2563eb")
    ax.set_title("Accuracy@1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(ep, [h["val_rc"] for h in history], label="val RC", color="#0ea5e9")
    ax.plot(ep, [h["val_spike"] for h in history], label="val spike", color="#ea580c")
    ax.set_title("RC / spike (val)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(ep, [h["max_grad_l2"] for h in history], label="max grad L2", color="#ef4444")
    ax.set_title("Gradient norm (max so far)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp08 addon: predict/memorize measurable test")
    ap.add_argument("--max-vocab", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--mem-batches", type=int, default=2, help="кол-во батчей в memorization holdout")
    ap.add_argument("--val-batches", type=int, default=6)
    ap.add_argument("--test-batches", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--minutes", type=int, default=0, help="цель по времени (приближённо): увеличивает epochs")
    ap.add_argument(
        "--memorize-hard",
        action="store_true",
        help="жёсткий меморизейшн: 1 фиксированное train-окно (B=1) + много повторов",
    )
    ap.add_argument("--mem-start", type=int, default=123456, help="start offset для фиксированного train окна")
    ap.add_argument("--no-rfp", action="store_true", help="только Adam (для чистого memorization)")
    ap.add_argument("--rfp-interval", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mem-seed", type=int, default=111)
    ap.add_argument("--val-seed", type=int, default=4242)
    ap.add_argument("--test-seed", type=int, default=7777)
    ap.add_argument(
        "--pos-mode",
        type=str,
        default="absolute",
        choices=("absolute", "token_as_pos"),
        help="как формировать positions для WPE",
    )
    ap.add_argument("--sample-top-k", type=int, default=20)
    ap.add_argument("--sample-temp", type=float, default=0.9)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    corpus = WikiTextCharCorpus.from_hf_disk(max_vocab=args.max_vocab)
    out_dir = _EXP / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.save_vocab_json(out_dir / "wikitext08_vocab_meta.json")

    epochs = args.epochs
    if args.quick:
        epochs = min(epochs, 12)
    if args.minutes and args.minutes >= 5:
        epochs = max(epochs, 120)

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
    model = WFRLM(core, vocab_size=corpus.vocab_size, num_phases=16).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # фиксированные наборы батчей:
    # - mem: либо фиксированное одно train-окно (memorize-hard), либо несколько фиксированных train-батчей
    # - val/test: фиксированные holdout из соответствующих сплитов
    if args.memorize_hard:
        mem_x = corpus.fixed_train_window(start=args.mem_start, seq_len=args.seq_len, device=DEVICE)
        mem_batches = [mem_x]  # [1, seq]
        mem_train_repeats = max(8, args.mem_batches * 10)
    else:
        g_mem = torch.Generator()
        g_mem.manual_seed(args.mem_seed)
        mem_batches = [
            corpus.sample_train_batch(args.batch_size, args.seq_len, g_mem, DEVICE)
            for _ in range(args.mem_batches)
        ]
        mem_train_repeats = 1

    val_batches = corpus.make_val_batches(
        args.val_batches, args.batch_size, args.seq_len, args.val_seed, DEVICE
    )
    test_batches = corpus.make_test_batches(
        args.test_batches, args.batch_size, args.seq_len, args.test_seed, DEVICE
    )

    fp = {
        "mem_sha256": tensor_batches_fingerprint(mem_batches),
        "val_sha256": tensor_batches_fingerprint(val_batches),
        "test_sha256": tensor_batches_fingerprint(test_batches),
    }

    history: list[dict[str, float]] = []
    max_g = 0.0
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        # memorization phase: много раз гоняем один и тот же mem-holdout
        for _ in range(mem_train_repeats):
            for batch in mem_batches:
                global_step += 1
                opt.zero_grad(set_to_none=True)
                pos = make_positions(batch, pos_mode=args.pos_mode)
                st = model(pos, batch)
                ce = F.cross_entropy(
                    st.logits[:, :-1].reshape(-1, st.logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
                rc_term = 1.0 - st.rc if st.rc.dim() == 0 else (1.0 - st.rc).mean()
                total = ALPHA * ce + BETA * rc_term + GAMMA * st.energy
                total.backward()
                g2 = total_grad_l2_norm(model)
                max_g = max(max_g, g2)
                opt.step()

                if (not args.no_rfp) and (global_step % args.rfp_interval == 0):
                    deltas = rfp_step_v03(
                        model.core,
                        st,
                        ce.detach(),
                        spike_target=0.25,
                        eta_f=1.2e-4,
                        eta_theta=3e-4,
                        eta_alpha=8e-6,
                        rescue_threshold=0.12,
                        rescue_factor=0.45,
                    )
                    apply_rfp_deltas(model.core, deltas)

        # eval: та же схема positions
        def _eval_with_mode(bs: list[torch.Tensor]) -> EvalMetrics:
            model.eval()
            ce_sum = acc_sum = rc_sum = e_sum = sp_sum = 0.0
            n_batches = 0
            for x in bs:
                pos = make_positions(x, pos_mode=args.pos_mode)
                st = model(pos, x)
                logits = st.logits[:, :-1]
                tgt = x[:, 1:]
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                pred = logits.argmax(dim=-1)
                acc = (pred == tgt).float().mean()
                rc = st.rc if st.rc.dim() == 0 else st.rc.mean()
                e = st.energy if st.energy.dim() == 0 else st.energy.mean()
                sp = st.spikes.mean()
                ce_sum += float(ce.item())
                acc_sum += float(acc.item())
                rc_sum += float(rc.item())
                e_sum += float(e.item())
                sp_sum += float(sp.item())
                n_batches += 1
            inv = 1.0 / max(1, n_batches)
            return EvalMetrics(
                ce=ce_sum * inv,
                acc1=acc_sum * inv,
                rc=rc_sum * inv,
                energy=e_sum * inv,
                spike_rate=sp_sum * inv,
            )

        mem = _eval_with_mode(mem_batches)
        val = _eval_with_mode(val_batches)
        tes = _eval_with_mode(test_batches)

        history.append(
            {
                "epoch": float(ep),
                "mem_ce": mem.ce,
                "mem_acc1": mem.acc1,
                "val_ce": val.ce,
                "val_acc1": val.acc1,
                "test_ce": tes.ce,
                "test_acc1": tes.acc1,
                "val_rc": val.rc,
                "val_spike": val.spike_rate,
                "max_grad_l2": max_g,
            }
        )

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(
                f"ep {ep:4d} | mem acc@1 {mem.acc1:6.3f} ce {mem.ce:6.3f} | "
                f"val acc@1 {val.acc1:6.3f} ce {val.ce:6.3f} | "
                f"test acc@1 {tes.acc1:6.3f} ce {tes.ce:6.3f}"
            )

        # ранняя остановка для memorization-части: если mem acc уже очень высокая
        if args.quick and mem.acc1 > 0.92 and ep >= 6:
            break

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "baseline" if args.no_rfp else "rfp_v03"
    tag = f"predict_mem_{mode}_{ts}"
    out_json = out_dir / f"{tag}.json"
    out_png = out_dir / f"{tag}.png"
    out_txt = out_dir / f"{tag}_samples.txt"

    plot_curves(history, out_png, corpus.vocab_size)

    # samples: фиксированный префикс из val (первые символы) + продолжение
    prefix = corpus.val_text[:200]
    sample = generate(
        model,
        corpus,
        prefix=prefix,
        gen_len=300,
        temperature=args.sample_temp,
        top_k=args.sample_top_k,
    )
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("PREFIX (val[:200])\n")
        f.write(prefix)
        f.write("\n\nGREEDY CONTINUATION (prefix + 300)\n")
        f.write(sample)

    report: dict[str, Any] = {
        "when": ts,
        "device": str(DEVICE),
        "mode": mode,
        "vocab_size": corpus.vocab_size,
        "ln_vocab": math.log(corpus.vocab_size),
        "hparams": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs_ran": len(history),
            "lr": args.lr,
            "seed": args.seed,
            "mem_seed": args.mem_seed,
            "val_seed": args.val_seed,
            "test_seed": args.test_seed,
            "mem_batches": args.mem_batches,
            "val_batches": args.val_batches,
            "test_batches": args.test_batches,
        },
        "fingerprints": fp,
        "final": history[-1] if history else None,
        "history": history,
        "artifacts": {
            "json": out_json.name,
            "png": out_png.name,
            "samples_txt": out_txt.name,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"final": report["final"], "artifacts": report["artifacts"], "fingerprints": fp},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
