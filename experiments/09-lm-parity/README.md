# Experiment 09 — LM parity (Transformer baseline на том же протоколе, что WFR)

**Цель этапа A дорожной карты:** [`docs/12-wfr-llm-breakthrough-roadmap.md`](../../docs/12-wfr-llm-breakthrough-roadmap.md). **План узких мест / прорывов и регрессия ядра:** [`docs/14-core-readiness-and-breakthrough-matrix.md`](../../docs/14-core-readiness-and-breakthrough-matrix.md).

Мы сравниваем **не «идеальный GPT»**, а **минимальный causal Transformer** на **char WikiText-2** с тем же протоколом семплирования окон, что и [`run_wikitext_train.py`](../08-wikitext-rfp/run_wikitext_train.py): `make_train_batches_for_epoch` + `make_val_batches` с `val_seed=4242`.

## Запуск

Из корня репозитория (нужен датасет, см. `data/hf/download_wikitext2.py`):

```bash
pip install -e .
python experiments/09-lm-parity/run_transformer_char_baseline.py --quick
python experiments/09-lm-parity/run_transformer_char_baseline.py --epochs 24 --num-train-batches 10 --num-val-batches 6 --seq-len 96 --batch-size 12
```

### Парное сравнение A2 (Transformer vs WFRLM)

Тот же протокол окон; WFRLM: **только CE** (`loss_mode=ce_only`), **без RFP** — сопоставимо с objective у baseline.

```bash
python experiments/09-lm-parity/run_parity_pair.py --quick
# Рекомендуемый честный A2: match-capacity + AdamW/clip/LR как у Transformer:
python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity
# Расширенный readout WFRLM (дорожная карта B1); TF подгоняется под новый счёт параметров:
python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity --readout-feat-dim 16
# MLP readout (feat→hidden→vocab); `--match-capacity` пересчитывает TF под доп. параметры:
python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity --readout-feat-dim 16 --readout-mlp-hidden 64
# Локальный контент: вклад токена t−1 в фазы позиции t (см. docs/03-theory.md §9.1):
python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity --content-neighbor-mix
# Жёстче совпасть по n: --match-capacity-min-d-model 4 (риск узкого TF)
python experiments/09-lm-parity/run_parity_pair.py --epochs 24 --seq-len 96 --batch-size 12 --num-train-batches 10 --num-val-batches 6
python experiments/09-lm-parity/run_parity_pair.py --match-lr --tf-lr 3e-4
```

## Артефакты

- `outputs/transformer_baseline_*.json` — `best_val_ce`, метаданные модели, сиды.
- `outputs/parity_pair_*.json` — A2: оба прогона, `delta_best_val_ce_wfr_minus_transformer`, `num_trainable_params` у WFRLM (в манифесте пайплайна). При `--match-capacity` в `fairness_notes.capacity_match` — выбранная конфигурация TF и целевой счёт WFRLM.

## Удалённый GPU (A100), RULES.md §6

После `upload` минимальная проверка (шаг 9 в `docs/07-experiment-plan.md`): на сервере  
`bash ~/Desktop/WFR-Memory-Test/_remote_gpu_check.sh` — печатает `CUDA True/False`, имя GPU и запускает `run_parity_pair.py --quick --fair-parity`. Скрипт заливается вместе с деревом (`experiments/08-wikitext-rfp/remote_sync_wikitext.ps1`).

1. Задать `WFR_SSH_PASSWORD`, `WFR_SSH_HOSTKEY` (и при необходимости `WFR_SSH_HOST` / `WFR_SSH_USER`).
2. Загрузить дерево (включает Exp 09): из корня репозитория выполнить  
   `experiments\08-wikitext-rfp\remote_sync_wikitext.ps1 -Direction upload`
3. На сервере из `~/Desktop/WFR-Memory-Test`: датасет уже должен быть в `data/hf/wikitext-2-raw-v1/` (иначе `python data/hf/download_wikitext2.py` в venv).
4. **Сетка B1 readout × MLP** (после `upload`): на сервере  
   `sed -i 's/\r$//' experiments/09-lm-parity/remote_gpu_b1_mlp_matrix.sh; chmod +x experiments/09-lm-parity/remote_gpu_b1_mlp_matrix.sh`  
   затем `bash experiments/09-lm-parity/remote_gpu_b1_mlp_matrix.sh` (см. [`docs/14-core-readiness-and-breakthrough-matrix.md`](../../docs/14-core-readiness-and-breakthrough-matrix.md)).
5. **B3 combo** (`readout_wave_kernel=3` + MLP 64 при **D=32**), тот же бюджет, что parity:  
   `sed -i 's/\r$//' experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh && chmod +x experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh && bash experiments/09-lm-parity/remote_gpu_b3_combo_d32.sh`  
   или foreground один прогон:  
   `python experiments/09-lm-parity/run_parity_pair.py --fair-parity --epochs 48 --seq-len 96 --num-train-batches 20 --num-val-batches 8 --batch-size 16 --readout-feat-dim 32 --readout-mlp-hidden 64 --readout-wave-kernel 3`
6. **Этап D.1 (длины контекста, A100):** после `upload` нормализовать CRLF и запустить батч:  
   `sed -i 's/\r$//' experiments/09-lm-parity/remote_gpu_longcontext_d32.sh && chmod +x experiments/09-lm-parity/remote_gpu_longcontext_d32.sh && bash experiments/09-lm-parity/remote_gpu_longcontext_d32.sh`  
   Скрипт [`remote_gpu_longcontext_d32.sh`](remote_gpu_longcontext_d32.sh): пары длин (`LONGCONTEXT_SEQ_LENS`, по умолчанию `96,512`), **`PARITY_READOUT_DIM`** (по умолчанию **32**), те же `PARITY_EPOCHS` / `PARITY_NUM_TRAIN` / `PARITY_BATCH`, что у `remote_gpu_parity.sh`; опционально `PARITY_READOUT_MLP_HIDDEN`, `PARITY_PHASE_CAUSAL_KERNEL`, `PARITY_READOUT_WAVE_KERNEL`, `PARITY_CONTENT_NEIGHBOR_MIX=1`. Пишет лог в `outputs/remote_a100/runs/longcontext_d32_<ts>.log` и копирует свежий `longcontext_pair_*.json` в тот же каталог `runs/longcontext_d32_<ts>/`. При нехватке памяти на большой длине: **`PARITY_BATCH=8`**. Локально без GPU — тот же [`run_longcontext_pair.py`](run_longcontext_pair.py).

7. Длинный парный sweep `remote_gpu_parity.sh` (readout по `PARITY_READOUT_DIMS`, логи + JSON в `outputs/`):

```bash
chmod +x experiments/09-lm-parity/remote_gpu_parity.sh
# foreground:
bash experiments/09-lm-parity/remote_gpu_parity.sh
# фон (имя лога фиксированное — из PowerShell не подставляйте $(date), это съедает локальный shell):
nohup bash experiments/09-lm-parity/remote_gpu_parity.sh > experiments/09-lm-parity/outputs/parity_a100_nohup_outer.log 2>&1 &
```

Не запускайте второй раз, пока живёт реальный Python-прогон (см. `ps -eo args | grep -F run_parity_pair.py | grep -F .venv`). Обычный `pgrep -f run_parity_pair` может спутать с оболочкой `bash -c`, в строке которой есть то же имя файла.

Переопределение бюджета: `PARITY_EPOCHS`, `PARITY_SEQ_LEN`, `PARITY_NUM_TRAIN`, `PARITY_NUM_VAL`, `PARITY_BATCH`.  
Набор ширин readout (B1 sweep): `PARITY_READOUT_DIMS="3 8 16 32"` перед вызовом скрипта (по умолчанию `3 16`).  
Опционально ко всем прогонам sweep добавить MLP readout: `PARITY_READOUT_MLP_HIDDEN=64` (передаётся как `--readout-mlp-hidden`).  
Локальный контент: `PARITY_CONTENT_NEIGHBOR_MIX=1` → `--content-neighbor-mix`. Ядро: `PARITY_PHASE_CAUSAL_KERNEL=3` → `--phase-causal-kernel 3`. Readout-локальность по волне: `PARITY_READOUT_WAVE_KERNEL=3` → `--readout-wave-kernel 3` (см. `docs/03-theory.md` §9.1).

Парный локальный CLI:

```bash
python experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity --phase-causal-kernel 3
```

8. Скачать артефакты: `remote_sync_wikitext.ps1 -Direction download` — файлы попадут в **`outputs/remote_a100/`** (корень). Разнести по [`outputs/remote_a100/runs/`](outputs/remote_a100/README.md), чтобы не смешивать прогоны; описание каталогов — [`outputs/remote_a100/README.md`](outputs/remote_a100/README.md).

**Сводка по JSON:** `python experiments/09-lm-parity/summarize_parity_json.py` (по умолчанию только **sweep** `runs/02_sweep_readout_3_8_16_32_20260403`; второй каталог передавайте вручную, см. `outputs/remote_a100/README.md`).

### Подтверждение B3 на A100 (RULES.md §6)

Локальный `--quick` для `phase_causal_kernel` / `readout_wave_kernel` — только ориентир. Длинный бюджет (**48 ep**, seq 96, 20/8 батчей, bs 16), **`--fair-parity`**, **`readout_feat_dim=16`**, сиды **42/42**:

1. Загрузить дерево: `experiments\08-wikitext-rfp\remote_sync_wikitext.ps1 -Direction upload` (`WFR_SSH_*`).
2. На сервере:  
   `cd ~/Desktop/WFR-Memory-Test && chmod +x experiments/09-lm-parity/remote_gpu_b3_confirm.sh && bash experiments/09-lm-parity/remote_gpu_b3_confirm.sh`  
   или в фоне: `nohup bash experiments/09-lm-parity/remote_gpu_b3_confirm.sh > experiments/09-lm-parity/outputs/b3_confirm_nohup_outer.log 2>&1 &`
3. Логи: `outputs/b3_confirm_*.log`, JSON `parity_pair_*.json` в `outputs/`.
4. Скачать: `remote_sync_wikitext.ps1 -Direction download`, затем разнести артефакты в `outputs/remote_a100/runs/…` по аналогии с sweep.

Сводное описание этапа и теории — [`docs/13-project-status-snapshot.md`](../../docs/13-project-status-snapshot.md).

**Результаты A100 (`readout_feat_dim=16`, 48 ep, fair-parity, сиды 42/42) — 2026-04-03:**

| Конфиг | CE TF | CE WFR | Δ (WFR−TF) |
|--------|-------|--------|------------|
| Контроль | 3.113 | 3.160 | +0.047 |
| `--phase-causal-kernel 3` | 3.174 | 3.173 | −0.0015 |
| `--readout-wave-kernel 3` | 3.113 | 3.147 | +0.034 |

Каноничные JSON: [`outputs/remote_a100/runs/03_b3_confirm_20260403/`](outputs/remote_a100/runs/03_b3_confirm_20260403/).

**B1 — readout × MLP (`remote_gpu_b1_mlp_matrix.sh`), тот же 48 ep / fair-parity:** каталог [`outputs/remote_a100/runs/04_b1_mlp_matrix_20260403/`](outputs/remote_a100/runs/04_b1_mlp_matrix_20260403/).

| Конфиг | CE TF | CE WFR | Δ (WFR−TF) | ratio TF/WFR |
|--------|-------|--------|------------|--------------|
| D16 linear | 3.113 | 3.160 | +0.047 | 1.00 |
| D16 + MLP 64 | 2.886 | 3.018 | +0.132 | ~1.00 |
| D32 linear | 3.148 | 3.085 | **−0.063** | ~1.00 |
| D32 + MLP 64 | 3.090 | 2.988 | **−0.102** | ~1.00 |

Интерпретация: при **D=16** на этом прогоне MLP readout **ухудшил** Δ относительно линейной головы; при **D=32** MLP **усилил** преимущество WFR — см. **H2** в [`docs/14-core-readiness-and-breakthrough-matrix.md`](../../docs/14-core-readiness-and-breakthrough-matrix.md).

### Снято на A100 80GB (2026-04-03)

Один и тот же бюджет: **48** эпох, `seq_len=96`, 20 train / 8 val батчей, batch **16**, `--fair-parity`.

**Полный sweep** `readout_feat_dim ∈ {3,8,16,32}` — каталог [`outputs/remote_a100/runs/02_sweep_readout_3_8_16_32_20260403/`](outputs/remote_a100/runs/02_sweep_readout_3_8_16_32_20260403/) (JSON `122447`, `122623`, `122802`, `123043`).

| readout | ratio TF/WFR | CE TF | CE WFR | Δ (WFR−TF) |
|---------|-------------|-------|--------|------------|
| 3 | ~1.10 | 3.237 | 4.951 | +1.714 |
| 8 | ~1.01 | 3.156 | 3.948 | +0.792 |
| 16 | 1.00 | 3.113 | 3.160 | +0.047 |
| 32 | ~1.00 | 3.148 | 3.085 | **−0.063** |

**Отдельный прогон** только readout 3 и 16 (ранний батч до sweep) — [`runs/01_pair_readout_3_16_20260403/`](outputs/remote_a100/runs/01_pair_readout_3_16_20260403/) (`120311`, `120422`); метрики совпадают со строками 3 и 16 таблицы выше.

Вывод: узкий readout даёт большой зазор; с ростом ширины зазор сужается; при **readout 32** WFRLM **чуть лучше** baseline по best val CE при сопоставимом счёте параметров — подкрепление **B1**.

### Локальный smoke локальности B3 (CPU, 2026-04-03)

Один протокол: **`--quick --fair-parity`**, `readout_feat_dim=3`, сиды **`init/train=42`**. TinyTransformer подобран под число параметров WFR (≈5480 / строчка). **Короткие эпохи** — только ориентир; итоговые выводы по §A100 в таблице выше.

| Флаги | WFR params | JSON | CE TF | CE WFR | Δ (WFR−TF) |
|-------|------------|------|-------|--------|------------|
| baseline | 5214 | [`parity_pair_20260403_142152.json`](outputs/parity_pair_20260403_142152.json) | 5.720 | 5.799 | +0.078 |
| `--readout-wave-kernel 3` | 5217 | [`parity_pair_20260403_142218.json`](outputs/parity_pair_20260403_142218.json) | 5.720 | 5.778 | +0.058 |
| `--phase-causal-kernel 3` | 5262 | [`parity_pair_20260403_142446.json`](outputs/parity_pair_20260403_142446.json) | 5.720 | 5.473 | **−0.247** |
| `--content-neighbor-mix` | 5215 | [`parity_pair_20260403_142447.json`](outputs/parity_pair_20260403_142447.json) | 5.720 | 5.799 | +0.079 |

**Заметки:** свёртка по фазам (`pk3`) на этом микро-бюджете дала **отрицательный Δ** (WFRLM лучше matched TF) — нужно подтвердить на **48 ep / A100**. Смесь волны перед readout (`rw3`) слегка улучшила CE; сосед по контенту (`nb`) почти не отличилась от baseline.

### Локальный чеклист перед PR / GPU

- [`run_local_pr_checklist.ps1`](run_local_pr_checklist.ps1) — подряд: `compileall`, smoke, `pytest`, parity `--quick --fair-parity`, baseline TF, parity с `--readout-feat-dim 16`, опционально тесты стабильности и wikitext smoke, затем `verify_theory_calibration.py` и `verify_longcontext_artifacts.py`. См. [`docs/07-experiment-plan.md`](../../docs/07-experiment-plan.md) §10 шагов.

### Сверка теории с артефактами (локально, без GPU)

- [`summarize_parity_json.py`](summarize_parity_json.py) — таблица CE и Δ по каталогам с `parity_pair_*.json`.
- [`verify_theory_calibration.py`](verify_theory_calibration.py) — проверка **H1** (монотонное сужение зазора при росте `readout_feat_dim` в одном sweep) и сводка строк **B3** при наличии контроля/`phase_causal`/`readout_wave` на **D=16**. См. также `docs/03-theory.md` §9.2.
- [`verify_longcontext_artifacts.py`](verify_longcontext_artifacts.py) — структура JSON **`longcontext_pair_*.json`** (этап **D.1**): `runs[]`, `parity_payload`, `d1_metrics` (wall, peak CUDA). Полезно перед мерджем артефактов с GPU.

## Интерпретация

- В JSON A2 смотрите **`param_ratio_transformer_over_wfr`** и **`capacity_warning`**: по умолчанию TinyCausalTransformer (~10⁵ параметров) тяжелее WFRLM (~10³–10⁴). Тогда разница CE **не** отвечает на вопрос «какая архитектура лучше при равной мощности», пока вы не **сожмёте** Transformer вручную (например `--d-model 64 --nlayers 1 --dim-ff 256`) или не включите **`--match-capacity`**.
- По умолчанию `--match-capacity` держит **`--match-capacity-min-d-model 8`**, чтобы не сводить TF к `d_model=4`; при **`--match-capacity-min-d-model 4`** счёт ближе 1:1, но CE может упираться в оптимизацию. Полезно **`--match-lr`** и длиннее эпох до выводов.
- Если при **сопоставимых** параметрах и шагах **CE TF ≪ CE WFR** — приоритет **расширения readout / ширины WFR** (этап B), а не новых правил RFP.
- Если зазор небольшой — можно углублять масштаб и RFP (этапы C–E).
