"""
Live telemetry from wfr.core.WFRNetwork on synthetic positions.
Run: pip install -e . fastapi uvicorn
     python tools/visualizer/live_server.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from wfr.core import WFRNetwork

TARGET_POINTS = 256
DEFAULT_SEQ = 512


def _downsample_1d(t: torch.Tensor, n: int) -> list[float]:
    t = t.float().flatten()
    if t.numel() == 0:
        return []
    if t.numel() <= n:
        return t.tolist()
    idx = torch.linspace(0, t.numel() - 1, n, device=t.device).long()
    return t[idx].cpu().tolist()


def _downsample_phases(ph: torch.Tensor, n_pos: int) -> list[list[float]]:
    if ph.numel() == 0:
        return []
    s, m = ph.shape
    if s <= n_pos:
        rows = ph.cpu().tolist()
        return [[float(x) for x in row] for row in rows]
    idx = torch.linspace(0, s - 1, n_pos, device=ph.device).long()
    sub = ph[idx].cpu().tolist()
    return [[float(x) for x in row] for row in sub]


def build_telemetry(
    model: WFRNetwork,
    positions: torch.Tensor,
    context_length: int,
) -> dict:
    with torch.no_grad():
        out = model(positions)

    if positions.shape[0] == 0:
        raise RuntimeError("empty batch")

    rc = float(out["resonance_confidence"].item())
    sw = out["standing_wave"][0]
    standing_wave = _downsample_1d(sw, TARGET_POINTS)

    phases = out["phases"][0]
    phases_list = _downsample_phases(phases, min(128, phases.shape[0]))

    layers_out = []
    for i, resonance in enumerate(out["layer_resonances"]):
        r0 = resonance[0]
        sp0 = out["layer_spikes"][i][0]
        res_ds = _downsample_1d(r0, TARGET_POINTS)
        sp_ds = _downsample_1d(sp0, TARGET_POINTS)
        spikes_bool = [bool(v > 0.5) for v in sp_ds]
        spike_rate = float(sp0.mean().item())
        avg_amp = float(r0.abs().mean().item())
        layers_out.append(
            {
                "level": i,
                "mean_abs_res": avg_amp,
                "mean_spike": spike_rate,
                "resonances": [float(x) for x in res_ds],
                "spikes": spikes_bool,
                "spikeRate": spike_rate,
                "silentPct": float((1.0 - spike_rate) * 100.0),
                "avgAmplitude": avg_amp,
            }
        )

    return {
        "type": "telemetry",
        "contextLength": context_length,
        "seq_len": int(positions.shape[1]),
        "rc": rc,
        "layers": layers_out,
        "standingWave": standing_wave,
        "phases": phases_list,
        "standing_wave_mean_abs": float(sw.abs().mean().item()),
        "t": time.time(),
    }


def synth_positions(frame: int, seq_len: int, batch: int = 1) -> torch.Tensor:
    off = frame % 400
    base = torch.arange(1, seq_len + 1, dtype=torch.long) + off
    jitter = (frame * 7) % 13
    base = base + jitter
    base = base.clamp(min=1)
    pos_ok = base.unsqueeze(0).expand(batch, -1).clone()
    for b in range(1, batch):
        pos_ok[b] += b * 17
    return pos_ok


app = FastAPI(title="WFR Live Core Telemetry")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: WFRNetwork | None = None
_frame_counter = 0


def get_model() -> WFRNetwork:
    global _model
    if _model is None:
        _model = WFRNetwork(
            num_phases=16,
            num_fractal_levels=6,
            num_resonance_layers=4,
            homeostatic_enabled=False,
        )
        _model.train()
    return _model


@app.websocket("/ws")
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    global _frame_counter
    model = get_model()
    try:
        while True:
            _frame_counter += 1
            seq = DEFAULT_SEQ
            ctx_display = 2048
            pos = synth_positions(_frame_counter, seq_len=seq, batch=1)

            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(
                None,
                lambda p=pos: build_telemetry(model, p, ctx_display),
            )
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.12)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


@app.get("/health")
def health() -> dict:
    return {"ok": True, "core": "wfr.core.WFRNetwork"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
