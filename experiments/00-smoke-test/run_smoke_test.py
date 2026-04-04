"""
WFR-Architecture — Test 0: Smoke Test
======================================

Цель: проверить жизнеспособность базовой архитектуры ПО ТЕОРИИ.
Реализация: PhaseInterference + TheoreticalResonanceLayer.

Критерии успеха:
  1. Система запускается без ошибок
  2. WPE генерирует стабильные фазы
  3. Видны осмысленные паттерны стоячих волн
  4. Event-driven поведение наблюдается (не все нейроны активны)
  5. Resonance Confidence вычисляется
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wfr.core import WFRNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_smoke_test():
    print("=" * 60)
    print("WFR-Architecture — Test 0: Smoke Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # --- 1. Создаём сеть ---
    model = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=4,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Модель создана. Параметров: {total_params}")

    # --- 2. Создаём входные данные (позиции) ---
    seq_lengths = [512, 2048, 8192]
    batch_size = 1

    for seq_len in seq_lengths:
        print(f"\n--- Тест с контекстом {seq_len} ---")

        positions = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

        # --- 3. Forward pass ---
        with torch.no_grad():
            result = model(positions)

        phases = result["phases"]
        standing_wave = result["standing_wave"]
        rc = result["resonance_confidence"]
        layer_resonances = result["layer_resonances"]
        layer_spikes = result["layer_spikes"]
        
        # New theoretical diagnostics
        if "interferences" in result:
            interferences = result["interferences"]
            print(f"  Using Theoretical Resonance (Phase Interference + Resonance Function)")
        else:
            interferences = None

        # --- 4. Диагностика ---
        print(f"  Phases shape:        {phases.shape}")
        print(f"  Standing wave shape: {standing_wave.shape}")
        print(f"  Resonance Confidence: {rc.item():.4f}")

        for i, (res, spk) in enumerate(zip(layer_resonances, layer_spikes)):
            spike_rate = spk.mean().item()
            silent_pct = (1 - spike_rate) * 100
            avg_amp = res.abs().mean().item()
            print(f"  Layer {i}: spike_rate={spike_rate:.3f}  silent={silent_pct:.1f}%  avg_amplitude={avg_amp:.4f}")
            
            if interferences is not None:
                avg_interf = interferences[i].abs().mean().item()
                print(f"           interference_strength={avg_interf:.4f}")

        # --- 5. Визуализация ---
        visualize_results(seq_len, phases, standing_wave, layer_resonances, layer_spikes, rc, interferences)

    print("\n" + "=" * 60)
    print("Smoke Test завершён.")
    print(f"Графики сохранены в: {OUTPUT_DIR}")
    print("=" * 60)


def visualize_results(seq_len, phases, standing_wave, layer_resonances, layer_spikes, rc, interferences=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    title = f"WFR Smoke Test (THEORETICAL) — context={seq_len}, RC={rc.item():.4f}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Стоячая волна
    ax = axes[0, 0]
    sw = standing_wave[0].cpu().numpy()
    ax.plot(sw, linewidth=0.8, color="#22d3ee")
    ax.set_title("Standing Wave (средний резонанс по слоям)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.2)

    # 2. Фазовая карта (первые 8 фаз)
    ax = axes[0, 1]
    phase_data = phases[0, :min(512, phases.shape[1]), :8].cpu().numpy().T
    im = ax.imshow(phase_data, aspect="auto", cmap="twilight", interpolation="nearest")
    ax.set_title("Phase Map (первые 8 фаз)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Phase dim")
    plt.colorbar(im, ax=ax)

    # 3. Спайковая активность по слоям
    ax = axes[1, 0]
    spike_rates = [spk.mean().item() for spk in layer_spikes]
    silent_rates = [1 - sr for sr in spike_rates]
    x = range(len(spike_rates))
    ax.bar(x, spike_rates, color="#eab308", label="Active (spikes)")
    ax.bar(x, silent_rates, bottom=spike_rates, color="#374151", label="Silent")
    ax.set_title("Event-Driven Activity по слоям")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction")
    ax.legend()
    ax.set_ylim(0, 1)

    # 4. Резонансные амплитуды по слоям
    ax = axes[1, 1]
    for i, res in enumerate(layer_resonances):
        r = res[0].cpu().numpy()
        step = max(1, len(r) // 500)
        ax.plot(r[::step], linewidth=0.6, alpha=0.8, label=f"Layer {i}")
    ax.set_title("Resonance Amplitudes по слоям")
    ax.set_xlabel("Position")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"smoke_test_ctx{seq_len}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_smoke_test()
