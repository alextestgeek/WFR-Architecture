# WFR Visualizer

Интерактивный визуализатор архитектуры WFR (React + Three.js + Canvas 2D).

## Запуск

```bash
cd tools/visualizer
npm install
npm run dev
```

## Назначение

- Демонстрация стоячих волн и фазовой интерференции
- Визуализация спайковой активности по фрактальным уровням
- Интерактивное исследование параметров архитектуры

## Live: реальное ядро Python (`wfr.core`)

Терминал 1 (из корня репозитория, после `pip install -e .` и `pip install -e ".[viz]"` или `pip install fastapi uvicorn`):

```bash
python tools/visualizer/live_server.py
```

Терминал 2: `npm run dev`, в UI нажать **Live off** → **LIVE: Py core**. Поток ~8 Hz: синтетические `positions` [1..512]+сдвиг, forward через `WFRNetwork`, те же 2D-панели + 3D WPE → кольца слоёв → RC.

Переменная `VITE_WFR_LIVE_WS` (опционально) задаёт URL WebSocket.
