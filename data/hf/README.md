# Hugging Face — локальные датасеты для WFR

## `wikitext-2-raw-v1`

- **Карточка:** [wikitext](https://huggingface.co/datasets/wikitext) (конфиг `wikitext-2-raw-v1`).
- **Зачем:** небольшой реальный текст (~2–4 MB) для next-token / языкового fine-tuning без бенчмарка «как у GPT-4».
- **Сплиты:** `train`, `validation`, `test`.

Скачивание:

```powershell
pip install datasets
python data/hf/download_wikitext2.py
```

Артефакт: папка `data/hf/wikitext-2-raw-v1/` (`DatasetDict` от `datasets.save_to_disk`).
