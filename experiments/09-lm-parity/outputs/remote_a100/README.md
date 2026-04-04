# Артефакты A100 (Exp 09) — разделение по прогонам

Два **разных** запуска `remote_gpu_parity.sh` (одинаковый fair-parity бюджет: 48 ep, seq 96, 20/8 батчей, batch 16), чтобы не смешивать JSON и логи.

| Каталог | Что внутри |
|---------|------------|
| [`runs/01_pair_readout_3_16_20260403/`](runs/01_pair_readout_3_16_20260403/) | Только **`readout_feat_dim` 3 и 16** (дефолт скрипта до sweep). JSON: `parity_pair_20260403_120311.json`, `...120422.json`. |
| [`runs/02_sweep_readout_3_8_16_32_20260403/`](runs/02_sweep_readout_3_8_16_32_20260403/) | Полный sweep **`PARITY_READOUT_DIMS="3 8 16 32"`**, batch `TS=20260403_122424`. JSON: `122447` (3), `122623` (8), `122802` (16), `123043` (32). Полный лог: `parity_a100_sweep_nohup.log`. |
| [`runs/03_b3_confirm_20260403/`](runs/03_b3_confirm_20260403/) | **`remote_gpu_b3_confirm.sh`**: при `readout_feat_dim=16`, **48 ep**, fair-parity, сиды 42/42. JSON: `parity_pair_20260403_151528` (контроль), `151630` (`--phase-causal-kernel 3`), `151732` (`--readout-wave-kernel 3`). В корне `remote_a100/` дубли тех же файлов + `b3_confirm_*.log`. |
| [`runs/04_b1_mlp_matrix_20260403/`](runs/04_b1_mlp_matrix_20260403/) | **`remote_gpu_b1_mlp_matrix.sh`**: readout **16 / 32** × (linear + `mlp64`), **48 ep**, fair-parity, тот же бюджет батчей, что sweep. JSON: `153505` (D16 linear), `153604` (D16+mlp64), `153706` (D32 linear), `153813` (D32+mlp64). Лог: `b1_mlp_matrix_nohup_outer.log`. |
| `runs/longcontext_d32_<ts>/` | После **`remote_gpu_longcontext_d32.sh`**: лог `longcontext_d<ts>.log`, копия **`longcontext_pair_<ts>.json`** (этап D.1, несколько `--seq-lens`). |
| [`runs/p1_p2_p3_20260403/`](runs/p1_p2_p3_20260403/) | Цепочка **`_remote_pipeline_p1_p2_p3.sh`** (2026-04-03, A100): **P1** B3 combo D32+rw3+mlp64, **48 ep**, Δ≈−0.036 (WFR лучше); **P2** тот же combo **96 ep**, Δ≈+0.064 (TF лучше); **P3** `longcontext_pair` **96+512**, `PARITY_BATCH=8`, D=32, Δ≈−0.084 / −0.071, peak CUDA MiB ≈23.6 / 55.6. Файлы: `parity_pair_*_184828.json`, `*_185010.json`, `longcontext_pair_*_185208.json`, `pipeline_p1_p2_p3.log`. |

Дубликаты (`parity_pair_*` с теми же метриками и тем же readout, но другим временем в имени) **удалены** из репозитория.

**Сводка по JSON:** из корня Exp 09:

```bash
python experiments/09-lm-parity/summarize_parity_json.py
# вместе с ранним прогоном 01 (дубли 3 и 16 по метрикам совпадут со sweep):
python experiments/09-lm-parity/summarize_parity_json.py \
  experiments/09-lm-parity/outputs/remote_a100/runs/02_sweep_readout_3_8_16_32_20260403 \
  experiments/09-lm-parity/outputs/remote_a100/runs/01_pair_readout_3_16_20260403

# только B1 (readout × MLP):
python experiments/09-lm-parity/summarize_parity_json.py \
  experiments/09-lm-parity/outputs/remote_a100/runs/04_b1_mlp_matrix_20260403
```

После следующего `remote_sync_wikitext.ps1 -Direction download` новые файлы снова попадут в **корень** `remote_a100/` — перенесите нужные в новый `runs/NN_...` или удалите дубликаты вручную.
