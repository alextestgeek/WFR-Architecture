"""
WFR Layer Scaling Test — Experiment 04
=======================================

Цель: проверить, решают ли механизмы стабилизации v2.0
(Phase-Locking, Homeostatic, Surrogate Gradient) проблему
"мёртвых" слоёв при масштабировании до 32 фрактальных уровней.

Стратегии распределения частот:
  A) linear     — текущая (baseline): f = 1.0 + i * 0.5
  B) logarithmic — f = 2^(i / (N/4)), мягкий лог. рост
  C) harmonic   — f = (i+1) * f_base, целочисленные гармоники
  D) constant   — f = 1.0 для всех, вся дифференциация через homeostatic

Для порогов: единый начальный порог θ₀ = 0.3 для всех слоёв,
homeostatic regulation адаптирует индивидуально.
"""

import torch
import time
import math
import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).resolve().parent
_smoke = _root.parent / "00-smoke-test"
if _smoke.is_dir():
    sys.path.insert(0, str(_smoke))
else:
    sys.path.insert(0, str(_root))
from wfr_core import WFRNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LAYER_COUNTS = [4, 8, 16, 24, 32]
CONTEXT_SIZES = [512, 8192, 131072]
HOMEOSTATIC_WARMUP_PASSES = 200


def make_frequencies(strategy: str, n_layers: int) -> list:
    if strategy == "linear":
        return [1.0 + i * 0.5 for i in range(n_layers)]
    elif strategy == "logarithmic":
        return [2.0 ** (i / max(n_layers / 4, 1)) for i in range(n_layers)]
    elif strategy == "harmonic":
        f_base = 0.25
        return [f_base * (i + 1) for i in range(n_layers)]
    elif strategy == "constant":
        return [1.0] * n_layers
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def make_thresholds(n_layers: int, initial: float = 0.3) -> list:
    return [initial] * n_layers


STRATEGIES = ["linear", "logarithmic", "harmonic", "constant"]


def run_single_test(n_layers: int, strategy: str, context: int,
                    warmup_passes: int = HOMEOSTATIC_WARMUP_PASSES):
    freqs = make_frequencies(strategy, n_layers)
    thresholds = make_thresholds(n_layers)

    model = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=n_layers,
        layer_frequencies=freqs,
        layer_thresholds=thresholds,
        homeostatic_enabled=True,
        spike_rate_target=0.10,
        homeostatic_eta=0.01,
    ).to(DEVICE)
    model.target_mode = "frequency"
    model.eval()

    positions = torch.arange(context, device=DEVICE).unsqueeze(0)

    # Homeostatic warmup: прогоняем forward несколько раз,
    # чтобы пороги адаптировались к реальным данным
    for _ in range(warmup_passes):
        with torch.no_grad():
            model(positions)

    # Основной замер
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

    start = time.time()
    with torch.no_grad():
        output = model(positions)
    elapsed = time.time() - start

    if DEVICE.type == "cuda":
        mem_after = torch.cuda.max_memory_allocated()
        mem_used = mem_after - mem_before
    else:
        mem_used = 0

    rc = output["resonance_confidence"].item()

    per_layer = []
    for i, (res, spk) in enumerate(zip(output["layer_resonances"], output["layer_spikes"])):
        spike_rate = spk.mean().item()
        avg_amp = res.abs().mean().item()
        threshold_val = model.resonance_layers[i].spike_threshold.item()
        per_layer.append({
            "layer": i,
            "spike_rate": round(spike_rate, 4),
            "avg_amplitude": round(avg_amp, 6),
            "threshold_after_homeostatic": round(threshold_val, 4),
            "frequency": round(model.resonance_layers[i].frequency.item(), 4),
            "active": spike_rate > 0.001,
        })

    active_count = sum(1 for l in per_layer if l["active"])

    return {
        "n_layers": n_layers,
        "strategy": strategy,
        "context": context,
        "rc": round(rc, 4),
        "time_sec": round(elapsed, 4),
        "memory_bytes": int(mem_used),
        "active_layers": active_count,
        "total_layers": n_layers,
        "active_ratio": round(active_count / n_layers, 3),
        "warmup_passes": warmup_passes,
        "per_layer": per_layer,
    }


def run_full_experiment():
    print(f"WFR Layer Scaling Test")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Layer counts: {LAYER_COUNTS}")
    print(f"Contexts: {CONTEXT_SIZES}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Homeostatic warmup: {HOMEOSTATIC_WARMUP_PASSES} passes")
    print("=" * 80)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "N/A",
        "layer_counts": LAYER_COUNTS,
        "contexts": CONTEXT_SIZES,
        "strategies": STRATEGIES,
        "warmup_passes": HOMEOSTATIC_WARMUP_PASSES,
        "tests": [],
    }

    total = len(LAYER_COUNTS) * len(STRATEGIES) * len(CONTEXT_SIZES)
    idx = 0

    for n_layers in LAYER_COUNTS:
        for strategy in STRATEGIES:
            for context in CONTEXT_SIZES:
                idx += 1
                print(f"\n[{idx}/{total}] layers={n_layers} strategy={strategy} context={context:,}")

                try:
                    result = run_single_test(n_layers, strategy, context)
                    all_results["tests"].append(result)

                    print(f"  RC={result['rc']:.4f}  "
                          f"Active={result['active_layers']}/{n_layers} "
                          f"({result['active_ratio']*100:.0f}%)  "
                          f"Time={result['time_sec']:.4f}s")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results["tests"].append({
                        "n_layers": n_layers,
                        "strategy": strategy,
                        "context": context,
                        "error": str(e),
                    })

    save_results(all_results)
    plot_results(all_results)

    print_summary(all_results)
    return all_results


def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = OUTPUT_DIR / f"layer_scaling_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {path}")


def plot_results(results):
    tests = [t for t in results["tests"] if "error" not in t]
    if not tests:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("WFR Layer Scaling Test — v2.0 Stability Mechanisms", fontsize=14, fontweight="bold")

    colors = {"linear": "#ef4444", "logarithmic": "#3b82f6", "harmonic": "#22c55e", "constant": "#a855f7"}

    # 1. Active layer ratio vs layer count (context=8192)
    ax = axes[0, 0]
    for strat in STRATEGIES:
        subset = [t for t in tests if t["strategy"] == strat and t["context"] == 8192]
        if subset:
            xs = [t["n_layers"] for t in subset]
            ys = [t["active_ratio"] * 100 for t in subset]
            ax.plot(xs, ys, "o-", color=colors[strat], label=strat, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Active Layers (%)")
    ax.set_title("Layer Utilization (context=8192)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # 2. RC vs layer count (context=8192)
    ax = axes[0, 1]
    for strat in STRATEGIES:
        subset = [t for t in tests if t["strategy"] == strat and t["context"] == 8192]
        if subset:
            xs = [t["n_layers"] for t in subset]
            ys = [t["rc"] for t in subset]
            ax.plot(xs, ys, "o-", color=colors[strat], label=strat, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Resonance Confidence")
    ax.set_title("RC vs Layer Count (context=8192)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Per-layer spike rate heatmap (best strategy, 32 layers, context=8192)
    ax = axes[1, 0]
    best_32 = [t for t in tests if t["n_layers"] == 32 and t["context"] == 8192]
    if best_32:
        heatmap_data = []
        ylabels = []
        for t in best_32:
            rates = [l["spike_rate"] for l in t["per_layer"]]
            heatmap_data.append(rates)
            ylabels.append(t["strategy"])
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Layer Index")
        ax.set_title("Spike Rate per Layer (32 layers, ctx=8192)")
        plt.colorbar(im, ax=ax)

    # 4. Threshold adaptation (32 layers, context=8192)
    ax = axes[1, 1]
    for t in best_32:
        thresholds = [l["threshold_after_homeostatic"] for l in t["per_layer"]]
        ax.plot(thresholds, "o-", color=colors[t["strategy"]], label=t["strategy"],
                linewidth=1.5, markersize=4)
    ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="initial θ=0.3")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Threshold after Homeostatic")
    ax.set_title("Threshold Adaptation (32 layers, ctx=8192)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "layer_scaling.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plots saved: {path}")


def print_summary(results):
    tests = [t for t in results["tests"] if "error" not in t]
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Layers':>6} {'Strategy':>13} {'Context':>9} {'RC':>7} {'Active':>10} {'Ratio':>7} {'Time':>8}")
    print("-" * 80)

    for t in tests:
        print(f"{t['n_layers']:>6} {t['strategy']:>13} {t['context']:>9,} "
              f"{t['rc']:>7.4f} {t['active_layers']:>4}/{t['total_layers']:<4} "
              f"{t['active_ratio']*100:>5.0f}%  {t['time_sec']:>7.3f}s")

    # Best strategy for 32 layers
    best_32 = [t for t in tests if t["n_layers"] == 32]
    if best_32:
        print("\n" + "-" * 40)
        print("Best for 32 layers:")
        best = max(best_32, key=lambda t: t["active_ratio"])
        print(f"  Strategy: {best['strategy']}")
        print(f"  Active: {best['active_layers']}/{best['total_layers']} ({best['active_ratio']*100:.0f}%)")
        print(f"  RC: {best['rc']:.4f}")


if __name__ == "__main__":
    run_full_experiment()
    print("\nLayer Scaling Test complete.")
