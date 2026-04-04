# Runbook: следующие фазы после проверки U1–U6

**Перед загрузкой на GPU:** локально выполнить [`run_local_pr_checklist.ps1`](run_local_pr_checklist.ps1) (чеклист из `docs/07-experiment-plan.md`).

Смысловой план: [`docs/16-next-phases-verified.md`](../../docs/16-next-phases-verified.md). Удалённый GPU: [`RULES.md`](../../RULES.md) §6, синк — `experiments/08-wikitext-rfp/remote_sync_wikitext.ps1`.

## P1 — B3 combo + повтор H2

```bash
# На сервере, из ~/Desktop/WFR-Memory-Test после upload:
sed -i 's/\r$//' experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh && chmod +x experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh
bash experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh
```

Повтор матрицы B1 с другим seed: задать в `run_parity_pair.py` / окружении `PYTHONHASHSEED` и зафиксировать в JSON `init_seed` (если уже есть — дублировать прогон).

## P2 — Удлинение train (этап C)

```bash
export PARITY_EPOCHS=96   # пример
export PARITY_READOUT_DIMS=32
bash experiments/09-lm-parity/remote_gpu_parity.sh
```

См. переменные в [`README.md`](README.md) §удалённый GPU.

## P3 — Этап D (пары длин контекста)

**На A100 (рекомендуется):** после `upload` и `sed -i 's/\r$//'`:

```bash
chmod +x experiments/09-lm-parity/remote_gpu_longcontext_d32.sh
bash experiments/09-lm-parity/remote_gpu_longcontext_d32.sh
```

По умолчанию: `LONGCONTEXT_SEQ_LENS=96,512`, `PARITY_READOUT_DIM=32`, бюджет как у parity (`PARITY_EPOCHS=48`, …). При OOM: `PARITY_BATCH=8 bash …`. JSON копируется в `outputs/remote_a100/runs/longcontext_d32_<ts>/`.

**Локально / один в один CLI:**

```bash
python experiments/09-lm-parity/run_longcontext_pair.py \
  --fair-parity --seq-lens 96,512 \
  --epochs 48 --num-train-batches 20 --num-val-batches 8 --batch-size 16 \
  --readout-feat-dim 32
```

Артефакт: `outputs/longcontext_pair_<ts>.json`.

Проверка полей (локально): `python experiments/09-lm-parity/verify_longcontext_artifacts.py`.

## P4 — B2 (BPE)

Очередь: отдельная задача — загрузчик subword + parity с match-capacity.

## P5 — RFP (E)

После закрытия P1–P3 по критериям в `docs/12` §E.
