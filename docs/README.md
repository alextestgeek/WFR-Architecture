# WFR Documentation

**Wave-Fractal-Resonant Architecture**

**Версия:** 0.6  
**Дата:** 31 марта 2026

Это официальная папка документации проекта WFR.

---

## Навигация

| Файл                        | Назначение                                      | Статус |
|----------------------------|--------------------------------------------------|--------|
| [`00-overview.md`](00-overview.md)     | **Главный обзор всей архитектуры**              | **Начать здесь** |
| [`01-introduction.md`](01-introduction.md) | Философия и происхождение идеи                   | — |
| [`02-architecture.md`](02-architecture.md) | Детальное описание компонентов                   | — |
| [`03-theory.md`](03-theory.md)         | Математика, стабильность v2.1, Layer Scaling, **§10** токены, **§11** узкие места LM и вводные в ядро | Основной технический документ |
| [`04-comparison.md`](04-comparison.md) | Сравнение с другими архитектурами                | Обновлено (O(n) вместо O(log n)) |
| [`05-roadmap.md`](05-roadmap.md)       | План развития проекта                            | — |
| [`06-visualizer.md`](06-visualizer.md) | Описание интерактивного визуализатора            | — |
| [`07-experiment-plan.md`](07-experiment-plan.md) | План и статус экспериментов; **Phase 3+** четыре опоры; **чеклист из 10 шагов** перед PR (`core` / `wfr_lm`) | v0.6, 2026-04-03 |
| [`08-phase-0-plan.md`](08-phase-0-plan.md) | **Мастер-план Phase 0** (viability, критерии выхода) | Phase 0 закрыт |
| [`09-memory-complexity-test-plan.md`](09-memory-complexity-test-plan.md) | Результаты теста памяти (100M токенов) | Завершён |
| [`10-phase-1-plan.md`](10-phase-1-plan.md) | **Phase 1 — обучение, RFP**, критерии, оговорки | Phase 1 закрыт; Phase 2 в работе |
| [`11-rfp-v0-spec.md`](11-rfp-v0-spec.md) | **RFP v0** — интерфейсы, ядро, A/B метрики | Реализовано (Exp 06) |
| [`12-wfr-llm-breakthrough-roadmap.md`](12-wfr-llm-breakthrough-roadmap.md) | **Дорожная карта к прорыву в LLM** (теория + этапы A–F) | Активный план; этап A = Exp 09 |
| [`13-project-status-snapshot.md`](13-project-status-snapshot.md) | **Срез состояния:** доказано / гипотезы / LM-прорыв | Обновлять по этапам B–C |
| [`14-core-readiness-and-breakthrough-matrix.md`](14-core-readiness-and-breakthrough-matrix.md) | **Ядро: регрессия** + **матрица узких мест / гипотез** (H1–H8) | Ведение §3 после каждого снимка A100 |
| [`16-next-phases-verified.md`](16-next-phases-verified.md) | **Проверка тезисов U1–U6** + **план этапов P1–P5** (после parity) | Активный runbook-уровень |
| [`strategy-fractal-layers.md`](strategy-fractal-layers.md) | Стратегия параметров (4 слоя)         | Частично устарел (см. exp.04) |

---

## Рекомендуемый порядок изучения

1. `00-overview.md` — получить общее понимание
2. `01-introduction.md` — понять философию
3. `08-phase-0-plan.md` — цели Phase 0 и критерии завершения
4. `03-theory.md` — изучить математику, стабильность и результаты Layer Scaling
5. `09-memory-complexity-test-plan.md` — данные экспериментов
6. `10-phase-1-plan.md` — Phase 1 (обучение, RFP), если переходите от Phase 0  
7. `11-rfp-v0-spec.md` — спецификация и метрики RFP v0 (Experiment 06)
8. `12-wfr-llm-breakthrough-roadmap.md` — путь к прорывной LLM на WFR (parity → масштаб)
9. `13-project-status-snapshot.md`, `14-core-readiness-and-breakthrough-matrix.md`, `16-next-phases-verified.md` — срез, матрица H*, следующие фазы после проверки тезисов
10. Остальные документы — по мере необходимости

---

Документация обновляется по мере развития проекта.
