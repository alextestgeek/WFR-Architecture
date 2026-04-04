"""
Скачать wikitext-2-raw-v1 с Hugging Face и сохранить рядом (save_to_disk).

Запуск из корня репозитория:
  python data/hf/download_wikitext2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "wikitext-2-raw-v1"


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets", file=sys.stderr)
        raise SystemExit(1)

    print("Loading wikitext / wikitext-2-raw-v1 from Hugging Face Hub...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUT))
    print(f"Saved to {OUT}")
    print(ds)


if __name__ == "__main__":
    main()
