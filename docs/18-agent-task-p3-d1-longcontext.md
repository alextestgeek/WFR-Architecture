# 18. Задача для агента: этап P3 — D.1 long context (честный parity + память)

**ID:** `TASK-P3-D1`  
**Дата:** 4 апреля 2026  
**Связь:** [`16-next-phases-verified.md`](16-next-phases-verified.md) §3 P3, [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md) **H6**, [`12-wfr-llm-breakthrough-roadmap.md`](12-wfr-llm-breakthrough-roadmap.md) этап D.

---

## Зачем это нужно

Сейчас заявка WFR про **длинный контекст и память** слабее всего **измерена** на парных прогонах LM: в `16` и `14` прямо сказано, что **U6 / H6** требуют канонического отчёта D.1 (`96` vs `512` или другая пара) с **CE, wall time, peak CUDA** в одном JSON. Без этого агенты и люди продолжают спорить «на словах».

Это **не** замена ядру и **не** уход в обычный LLM: используется уже готовый [`run_longcontext_pair.py`](../experiments/09-lm-parity/run_longcontext_pair.py) + fair-parity.

---

## Ограничения (манифест)

Соблюдать [`17-agent-continuous-work-manifest.md`](17-agent-continuous-work-manifest.md): не добавлять self-attention в `wfr/core.py`; изменения только в скриптах запуска/доках/верификации, если не открыт отдельный RFC.

---

## Ветка A — полный прогон (есть GPU A100 / удалённый сервер)

1. Синк репозитория на сервер (как в `experiments/08-wikitext-rfp/remote_sync_wikitext.ps1` и `RULES.md` §6).
2. Нормализовать CRLF и права:
   ```bash
   sed -i 's/\r$//' experiments/09-lm-parity/remote_gpu_longcontext_d32.sh
   chmod +x experiments/09-lm-parity/remote_gpu_longcontext_d32.sh
   ```
3. Запуск по умолчанию (readout 32, seq `96,512`, 48 ep, те же батчи, что parity):
   ```bash
   bash experiments/09-lm-parity/remote_gpu_longcontext_d32.sh
   ```
   При OOM на длинной длине: `PARITY_BATCH=8 bash experiments/09-lm-parity/remote_gpu_longcontext_d32.sh`.
4. Скачать артефакты; положить JSON + лог в **`experiments/09-lm-parity/outputs/remote_a100/runs/<тег>/`** (как для других A100-прогонов, см. [`09-lm-parity/README`](../experiments/09-lm-parity/README.md)).
5. Локально проверить структуру:
   ```powershell
   python experiments/09-lm-parity/verify_longcontext_artifacts.py experiments/09-lm-parity/outputs/remote_a100/runs/<тег>
   ```
6. **Обновить документацию:**
   - [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md) §3: строка **H6** — статус, дата, путь к JSON; кратко: при росте `seq_len` как ведут себя Δ, `peak_cuda_mib` у WFR vs TF (из вложенных `parity_payload`).
   - [`13-project-status-snapshot.md`](13-project-status-snapshot.md) — абзац про D.1 / длинный контекст со ссылкой на `runs/…`.
   - При необходимости одна строка в [`experiments/09-lm-parity/README.md`](../experiments/09-lm-parity/README.md) в раздел про D.1.

### Критерий готовности ветки A

- Есть валидный `longcontext_pair_*.json` (verify exit 0).
- В §**14** обновлена **H6** с ссылкой на файл.
- В коммите видно **два** `seq_len` и для каждого — **delta_best_val_ce**, **peak_cuda_mib** (или явно «CPU», если прогон вынужденно без CUDA — тогда задача считается **черновиком**, не закрытием H6).

---

## Ветка B — только локально (нет GPU)

Цель: не блокировать работу; подготовить/проверить цепочку.

1. Убедиться, что скрипт стартует и пишет JSON:
   ```powershell
   cd c:\WFR-Architecture
   python experiments/09-lm-parity/run_longcontext_pair.py --quick --fair-parity --seq-lens 96,128
   ```
2. Запустить `verify_longcontext_artifacts.py` на каталоге `experiments/09-lm-parity/outputs` (или на свежем JSON).
3. Если найдены огрехи в README / `NEXT_PHASE_RUNBOOK.md` (неясные переменные окружения, шаг копирования в `runs/`) — **минимальный патч** + ссылка на эту задачу **18**.

Ветка B **не закрывает** H6 на таблице A100; в §14 помечать как «локальная регрессия скрипта / док» при отсутствии GPU-JSON.

---

## Отчёт агенту в конце сессии (обязательно)

Коротко, в коммите или в конце `07-experiment-plan.md`:

- Какая ветка (A или B).
- Путь к JSON / логу.
- Числа: по каждому `seq_len` — best val CE WFR/TF, Δ, peak MiB, wall s.
- Следующий шаг (например: повтор с другим сидом, `PARITY_EPOCHS=96`, или окно у TF — если появится отдельная гипотеза и RFC).

---

*После закрытия ветки A матрицу H6 ведите по правилу [`14` §4 п.6](14-core-readiness-and-breakthrough-matrix.md) (при споре — второй прогон).*
