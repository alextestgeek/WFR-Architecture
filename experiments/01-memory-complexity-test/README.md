# Experiment 01 — Memory & Complexity Test

**Статус:** Завершён (см. отчёт с таблицами и выводами по O(1) памяти).  
**Скрипт:** [`run_memory_test.py`](run_memory_test.py).

## Документация с результатами

Полный протокол, таблицы по контекстам до 100M токенов и сравнение v1.0 / v2.0 стека стабилизации:

- [`docs/09-memory-complexity-test-plan.md`](../../docs/09-memory-complexity-test-plan.md)

Мастер-план связывает этот тест с Phase 0:

- [`docs/07-experiment-plan.md`](../../docs/07-experiment-plan.md) (блок «Memory & Complexity Test»)

## Запуск

Из корня репозитория (нужен CUDA для замеров VRAM; иначе см. код/`--help`):

```bash
python experiments/01-memory-complexity-test/run_memory_test.py
```

Артефакты по умолчанию: `experiments/01-memory-complexity-test/outputs/memory_test_*.json` и логи рядом.

## Краткий итог

Память на токен остаётся **практически постоянной** при росте контекста; подробности и численные ряды — только в `docs/09-memory-complexity-test-plan.md`, чтобы не дублировать таблицы в двух местах.
