"""
Загрузка wikitext-2-raw-v1 с диска (HF save_to_disk) и char-level корпус для WFRLM.

Ожидаемый путь: ``<repo>/data/hf/wikitext-2-raw-v1/`` (см. ``data/hf/download_wikitext2.py``).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

def _repo_root() -> Path:
    """Корень репозитория или папка плоской копии на сервере (см. RULES.md §6)."""
    here = Path(__file__).resolve()
    if here.parent.name == "08-wikitext-rfp" and here.parent.parent.name == "experiments":
        return here.parents[2]
    return here.parent


ROOT = _repo_root()
DEFAULT_DATASET_DIR = ROOT / "data" / "hf" / "wikitext-2-raw-v1"


@dataclass(frozen=True)
class CharVocab:
    """Словарь символов: 0 = pad, 1 = unk, далее самые частые символы train."""

    char2id: dict[str, int]
    id2char: list[str]
    vocab_size: int

    def encode_slice(self, s: str) -> list[int]:
        unk = self.char2id["<unk>"]
        c2 = self.char2id
        return [c2.get(ch, unk) for ch in s]


def build_char_vocab(train_text: str, max_vocab: int = 256) -> CharVocab:
    if max_vocab < 8:
        raise ValueError("max_vocab must be >= 8")
    # резервируем pad + unk
    cap = max_vocab - 2
    counts = Counter(train_text)
    most = [ch for ch, _ in counts.most_common(cap)]
    char2id: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for i, ch in enumerate(most, start=2):
        char2id[ch] = i
    id2char = ["<pad>", "<unk>"] + most[: max_vocab - 2]
    # усечь id2char до фактического размера
    vs = len(id2char)
    return CharVocab(char2id=char2id, id2char=id2char, vocab_size=vs)


@dataclass
class WikiTextCharCorpus:
    """Текст train/val и методы сэмплирования окон фиксированной длины."""

    train_text: str
    val_text: str
    test_text: str
    vocab: CharVocab
    meta: dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return self.vocab.vocab_size

    @classmethod
    def from_hf_disk(
        cls,
        dataset_dir: Optional[Path] = None,
        *,
        max_vocab: int = 256,
    ) -> WikiTextCharCorpus:
        try:
            from datasets import load_from_disk
        except ImportError as e:
            raise RuntimeError("pip install datasets") from e

        path = Path(dataset_dir or DEFAULT_DATASET_DIR).resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Run from repo root: pip install datasets && python data/hf/download_wikitext2.py"
            )

        ds = load_from_disk(str(path))
        train_lines = [t for t in ds["train"]["text"] if isinstance(t, str) and t.strip()]
        val_lines = [t for t in ds["validation"]["text"] if isinstance(t, str) and t.strip()]
        test_lines = [t for t in ds["test"]["text"] if isinstance(t, str) and t.strip()]
        train_text = "\n".join(train_lines)
        val_text = "\n".join(val_lines)
        test_text = "\n".join(test_lines)
        if len(train_text) < 10_000:
            raise RuntimeError("Train text too short — check dataset path.")

        vocab = build_char_vocab(train_text, max_vocab=max_vocab)
        meta = {
            "dataset_dir": str(path),
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "test_chars": len(test_text),
            "train_lines": len(train_lines),
            "val_lines": len(val_lines),
            "test_lines": len(test_lines),
            "max_vocab_cap": max_vocab,
            "vocab_size": vocab.vocab_size,
        }
        return cls(
            train_text=train_text,
            val_text=val_text,
            test_text=test_text,
            vocab=vocab,
            meta=meta,
        )

    def save_vocab_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "id2char": self.vocab.id2char,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _window_tensor(
        self,
        text: str,
        starts: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """starts: [B] long CPU int64."""
        b = starts.shape[0]
        out = torch.empty(b, seq_len, dtype=torch.long, device=device)
        c2 = self.vocab.char2id
        unk = c2["<unk>"]
        for i in range(b):
            st = int(starts[i].item())
            chunk = text[st : st + seq_len]
            if len(chunk) < seq_len:
                # редкий край — паддинг нулями
                ids = [c2.get(ch, unk) for ch in chunk.ljust(seq_len)]
            else:
                ids = [c2.get(ch, unk) for ch in chunk]
            out[i] = torch.tensor(ids, dtype=torch.long, device=device)
        return out

    def sample_train_batch(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        text = self.train_text
        L = len(text)
        if L < seq_len + 16:
            raise ValueError("train_text shorter than seq_len")
        hi = L - seq_len
        # randint + Generator на CPU — стабильно на всех версиях PyTorch
        starts = torch.randint(0, hi, (batch_size,), generator=generator)
        return self._window_tensor(text, starts, seq_len, device)

    def make_val_batches(
        self,
        num_batches: int,
        batch_size: int,
        seq_len: int,
        seed: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Фиксированный holdout: детерминированные окна из val."""
        text = self.val_text
        L = len(text)
        if L < seq_len + 16:
            raise ValueError("val_text shorter than seq_len")
        hi = L - seq_len
        g = torch.Generator()
        g.manual_seed(seed)
        batches: list[torch.Tensor] = []
        for _ in range(num_batches):
            starts = torch.randint(0, hi, (batch_size,), generator=g)
            batches.append(self._window_tensor(text, starts, seq_len, device))
        return batches

    def make_test_batches(
        self,
        num_batches: int,
        batch_size: int,
        seq_len: int,
        seed: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Фиксированный holdout: детерминированные окна из test split."""
        text = self.test_text
        L = len(text)
        if L < seq_len + 16:
            raise ValueError("test_text shorter than seq_len")
        hi = L - seq_len
        g = torch.Generator()
        g.manual_seed(seed)
        batches: list[torch.Tensor] = []
        for _ in range(num_batches):
            starts = torch.randint(0, hi, (batch_size,), generator=g)
            batches.append(self._window_tensor(text, starts, seq_len, device))
        return batches

    def fixed_train_window(
        self,
        *,
        start: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Один фиксированный train-отрезок [1, seq_len] для memorization."""
        if start < 0:
            raise ValueError("start must be >= 0")
        text = self.train_text
        hi = len(text) - seq_len
        if hi <= 0:
            raise ValueError("train_text shorter than seq_len")
        st = min(start, hi)
        starts = torch.tensor([st], dtype=torch.long)
        return self._window_tensor(text, starts, seq_len, device)

    def make_train_batches_for_epoch(
        self,
        num_batches: int,
        batch_size: int,
        seq_len: int,
        epoch: int,
        train_seed: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Несколько train-батчей на эпоху (протокол fresh-train, как в protocol_train)."""
        g = torch.Generator()
        g.manual_seed(train_seed + epoch * 100_003)
        return [
            self.sample_train_batch(batch_size, seq_len, g, device) for _ in range(num_batches)
        ]
