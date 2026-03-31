"""
WFR Long Context Stability Test — Experiment 03
=================================================

Цель: проверить стабильность системы при росте длины контекста.

Что проверяем (чего НЕ проверял Memory & Complexity Test):

1. Детерминизм — один и тот же вход → один и тот же выход
2. Стабильность фазового кодирования — фазы позиции p одинаковы
   при любом контексте (архитектурная гарантия WPE)
3. Кросс-контекстная устойчивость стоячих волн — шаблон стоячей волны
   для позиций 0..511 одинаков при контекстах 512, 4096, 16384, 65536
4. RC по окнам — sliding window RC внутри длинного контекста
   (нет ли "мёртвых зон")
5. Распределение спайков по позициям — равномерно ли стреляют?
6. Фазовая когерентность по глубине последовательности —
   не деградирует ли RC к концу длинного контекста?
"""

import torch
import time
import json
import sys
import math
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = Path(__file__).resolve().parent
_smoke = _root.parent / "00-smoke-test"
if _smoke.is_dir():
    sys.path.insert(0, str(_smoke))
else:
    sys.path.insert(0, str(_root))
from wfr_core import WFRNetwork, ResonanceConfidence

# ==================== КОНФИГУРАЦИЯ ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CONTEXT_SIZES = [512, 4096, 16384, 65536]
HOMEOSTATIC_WARMUP = 50
DETERMINISM_RUNS = 3
WINDOW_SIZE = 512

BEST_CONFIG = {
    "target_mode": "frequency",
    "frequencies": [1.0, 1.8, 3.0, 5.0],
    "thresholds": [0.20, 0.25, 0.30, 0.40],
    "name": "Freq-Balanced",
}

print(f"Long Context Stability Test — Experiment 03")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Contexts: {CONTEXT_SIZES}")
print(f"Homeostatic warmup: {HOMEOSTATIC_WARMUP} passes")
print("=" * 80)


def build_model():
    model = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=4,
        layer_frequencies=BEST_CONFIG["frequencies"][:],
        layer_thresholds=BEST_CONFIG["thresholds"][:],
        homeostatic_enabled=True,
        spike_rate_target=0.10,
        homeostatic_eta=0.01,
    ).to(DEVICE)
    model.target_mode = BEST_CONFIG["target_mode"]
    model.eval()
    return model


def warmup_model(model, context_len):
    positions = torch.arange(context_len, device=DEVICE).unsqueeze(0)
    for _ in range(HOMEOSTATIC_WARMUP):
        with torch.no_grad():
            model(positions)


def compute_windowed_rc(phases, window_size):
    """RC по скользящим окнам внутри последовательности."""
    rc_calc = ResonanceConfidence()
    _, seq_len, _ = phases.shape
    n_windows = seq_len // window_size
    rcs = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window_phases = phases[:, start:end, :]
        rc_val = rc_calc(window_phases).item()
        rcs.append(rc_val)
    return rcs


def cosine_similarity(a, b):
    """Cosine similarity между двумя 1D тензорами."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dot = (a_flat * b_flat).sum()
    norm = a_flat.norm() * b_flat.norm()
    if norm < 1e-12:
        return 0.0
    return (dot / norm).item()


# ==================== ТЕСТЫ ====================


def test_1_determinism():
    """Один вход → идентичный выход при нескольких прогонах."""
    print("\n" + "=" * 60)
    print("TEST 1: Determinism")
    print("=" * 60)

    results = {}
    for ctx in CONTEXT_SIZES:
        print(f"\n  Context = {ctx:,}")
        positions = torch.arange(ctx, device=DEVICE).unsqueeze(0)

        phases_list = []
        rc_list = []
        spikes_list = []

        for run in range(DETERMINISM_RUNS):
            torch.manual_seed(42)
            model = build_model()
            warmup_model(model, ctx)

            with torch.no_grad():
                out = model(positions)
            phases_list.append(out["phases"].clone())
            rc_list.append(out["resonance_confidence"].item())
            total_spikes = sum(s.sum().item() for s in out["layer_spikes"])
            spikes_list.append(total_spikes)

        max_phase_diff = 0.0
        for i in range(1, DETERMINISM_RUNS):
            diff = (phases_list[0] - phases_list[i]).abs().max().item()
            max_phase_diff = max(max_phase_diff, diff)

        rc_spread = max(rc_list) - min(rc_list)
        spike_spread = max(spikes_list) - min(spikes_list)
        passed = max_phase_diff < 1e-5 and rc_spread < 1e-5

        results[ctx] = {
            "max_phase_diff": round(max_phase_diff, 10),
            "rc_values": [round(r, 6) for r in rc_list],
            "rc_spread": round(rc_spread, 10),
            "spike_counts": spikes_list,
            "spike_spread": spike_spread,
            "passed": passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"    max Δphase = {max_phase_diff:.2e}  |  RC spread = {rc_spread:.2e}  |  [{status}]")

    all_passed = all(r["passed"] for r in results.values())
    print(f"\n  Determinism: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return {"test": "determinism", "passed": all_passed, "details": results}


def test_2_phase_encoding_stability():
    """Фазы позиции p одинаковы при разных контекстах."""
    print("\n" + "=" * 60)
    print("TEST 2: Phase Encoding Stability")
    print("=" * 60)

    torch.manual_seed(42)
    encoder_model = build_model()

    probe_positions = [0, 10, 100, 511]
    reference_ctx = CONTEXT_SIZES[0]

    ref_positions = torch.arange(reference_ctx, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        ref_phases = encoder_model.encoder(ref_positions)

    results = {}
    for ctx in CONTEXT_SIZES[1:]:
        positions = torch.arange(ctx, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            phases = encoder_model.encoder(positions)

        ctx_results = {}
        for p in probe_positions:
            if p < reference_ctx and p < ctx:
                diff = (ref_phases[0, p, :] - phases[0, p, :]).abs().max().item()
                sim = cosine_similarity(ref_phases[0, p, :], phases[0, p, :])
                ctx_results[p] = {
                    "max_diff": round(diff, 10),
                    "cosine_sim": round(sim, 8),
                    "passed": diff < 1e-5,
                }
                print(f"  ctx={ctx:>6,}  pos={p:>4}  |  max Δ = {diff:.2e}  |  cos = {sim:.8f}")

        results[ctx] = ctx_results

    all_passed = all(
        v["passed"]
        for ctx_r in results.values()
        for v in ctx_r.values()
    )
    print(f"\n  Phase Encoding Stability: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return {"test": "phase_encoding_stability", "passed": all_passed, "details": results}


def test_3_cross_context_standing_wave():
    """Стоячая волна для общего префикса (0..511) при разных контекстах.

    Методология: ОДНА модель, warmup на опорном контексте (512),
    затем заморозка порогов и замеры на всех контекстах.
    WFR обрабатывает каждую позицию независимо, поэтому с
    фиксированными порогами стоячие волны ДОЛЖНЫ совпадать.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Cross-Context Standing Wave Stability")
    print("  (single model, warmup at ctx=512, frozen thresholds)")
    print("=" * 60)

    prefix_len = 512
    ref_ctx = CONTEXT_SIZES[0]

    torch.manual_seed(42)
    model = build_model()
    warmup_model(model, ref_ctx)

    for layer in model.resonance_layers:
        layer.homeostatic_enabled = False
    print(f"  Warmup done at ctx={ref_ctx}. Homeostatic frozen.")

    thresholds_after_warmup = []
    for i, layer in enumerate(model.resonance_layers):
        th = layer.spike_threshold.item()
        thresholds_after_warmup.append(round(th, 6))
        print(f"    Layer {i} threshold = {th:.6f}")

    standing_waves = {}
    for ctx in CONTEXT_SIZES:
        positions = torch.arange(ctx, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(positions)

        sw_prefix = out["standing_wave"][:, :prefix_len].clone()
        standing_waves[ctx] = sw_prefix

        rc = out["resonance_confidence"].item()
        print(f"  ctx={ctx:>6,}  |  RC = {rc:.4f}  |  SW prefix shape = {sw_prefix.shape}")

    ref_sw = standing_waves[ref_ctx]
    results = {"thresholds_after_warmup": thresholds_after_warmup}

    for ctx in CONTEXT_SIZES[1:]:
        sw = standing_waves[ctx]
        sim = cosine_similarity(ref_sw, sw)
        max_diff = (ref_sw - sw).abs().max().item()
        mean_diff = (ref_sw - sw).abs().mean().item()
        passed = sim > 0.95

        results[ctx] = {
            "cosine_sim_vs_512": round(sim, 6),
            "max_diff": round(max_diff, 6),
            "mean_diff": round(mean_diff, 6),
            "passed": passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"  ctx={ctx:>6,} vs {ref_ctx}  |  cos = {sim:.6f}  |  max Δ = {max_diff:.6f}  [{status}]")

    detail_results = {k: v for k, v in results.items() if isinstance(k, int)}
    all_passed = all(r["passed"] for r in detail_results.values())
    print(f"\n  Cross-Context Standing Wave: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return {"test": "cross_context_standing_wave", "passed": all_passed, "details": results}


def test_4_windowed_rc():
    """Sliding window RC — нет ли мёртвых зон внутри длинного контекста."""
    print("\n" + "=" * 60)
    print("TEST 4: Windowed RC (Dead Zone Detection)")
    print("=" * 60)

    results = {}

    for ctx in CONTEXT_SIZES:
        torch.manual_seed(42)
        model = build_model()
        warmup_model(model, ctx)

        positions = torch.arange(ctx, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(positions)

        window_rcs = compute_windowed_rc(out["phases"], WINDOW_SIZE)
        n_windows = len(window_rcs)

        if n_windows > 0:
            mean_rc = np.mean(window_rcs)
            std_rc = np.std(window_rcs)
            min_rc = np.min(window_rcs)
            max_rc = np.max(window_rcs)
            cv = std_rc / mean_rc if mean_rc > 1e-8 else float("inf")
            passed = cv < 0.15
        else:
            mean_rc = std_rc = min_rc = max_rc = cv = 0.0
            passed = True

        results[ctx] = {
            "n_windows": n_windows,
            "mean_rc": round(float(mean_rc), 6),
            "std_rc": round(float(std_rc), 6),
            "min_rc": round(float(min_rc), 6),
            "max_rc": round(float(max_rc), 6),
            "cv": round(float(cv), 6),
            "passed": passed,
            "window_rcs": [round(float(r), 6) for r in window_rcs],
        }
        status = "PASS" if passed else "FAIL"
        print(f"  ctx={ctx:>6,}  |  windows={n_windows:>4}  |  "
              f"RC: {mean_rc:.4f} ± {std_rc:.4f}  (CV={cv:.4f})  |  "
              f"range [{min_rc:.4f}, {max_rc:.4f}]  [{status}]")

    all_passed = all(r["passed"] for r in results.values())
    print(f"\n  Windowed RC: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return {"test": "windowed_rc", "passed": all_passed, "details": results}


def test_5_spike_distribution():
    """Равномерность распределения спайков по позициям.

    Методология: ОДНА модель, warmup на наибольшем контексте,
    заморозка порогов, затем замеры на всех контекстах.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Spike Distribution Uniformity")
    print("  (single model, warmup at largest ctx, frozen thresholds)")
    print("=" * 60)

    torch.manual_seed(42)
    model = build_model()
    warmup_ctx = CONTEXT_SIZES[-1]
    warmup_model(model, warmup_ctx)

    for layer in model.resonance_layers:
        layer.homeostatic_enabled = False
    print(f"  Warmup done at ctx={warmup_ctx:,}. Homeostatic frozen.")
    for i, layer in enumerate(model.resonance_layers):
        print(f"    Layer {i} threshold = {layer.spike_threshold.item():.6f}")

    results = {}

    for ctx in CONTEXT_SIZES:
        positions = torch.arange(ctx, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(positions)

        all_spikes = torch.stack(out["layer_spikes"], dim=0).sum(dim=0)  # [batch, seq]

        n_chunks = min(ctx // WINDOW_SIZE, 128)
        if n_chunks < 2:
            results[ctx] = {
                "n_chunks": n_chunks,
                "passed": True,
                "note": "too short for chunk analysis",
            }
            print(f"  ctx={ctx:>6,}  |  too short for chunk analysis  [SKIP]")
            continue

        chunk_size = ctx // n_chunks
        chunk_densities = []
        for c in range(n_chunks):
            start = c * chunk_size
            end = start + chunk_size
            density = all_spikes[0, start:end].mean().item()
            chunk_densities.append(density)

        mean_d = np.mean(chunk_densities)
        min_d = np.min(chunk_densities)
        max_d = np.max(chunk_densities)

        if min_d > 1e-8:
            ratio = max_d / min_d
        elif max_d < 1e-8:
            ratio = 1.0
        else:
            ratio = float("inf")

        dead_chunks = sum(1 for d in chunk_densities if d < 1e-6)
        passed = ratio < 3.0 and dead_chunks == 0

        per_layer_rates = []
        for i, spk in enumerate(out["layer_spikes"]):
            rate = spk.mean().item()
            per_layer_rates.append(round(rate, 6))

        results[ctx] = {
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "mean_density": round(float(mean_d), 6),
            "min_density": round(float(min_d), 6),
            "max_density": round(float(max_d), 6),
            "max_min_ratio": round(float(ratio), 4),
            "dead_chunks": dead_chunks,
            "per_layer_spike_rates": per_layer_rates,
            "passed": passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"  ctx={ctx:>6,}  |  chunks={n_chunks}  |  density: {mean_d:.4f}  "
              f"[{min_d:.4f}, {max_d:.4f}]  ratio={ratio:.2f}  dead={dead_chunks}  [{status}]")
        print(f"             |  per-layer rates: {per_layer_rates}")

    all_passed = all(r["passed"] for r in results.values())
    print(f"\n  Spike Distribution: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return {"test": "spike_distribution", "passed": all_passed, "details": results}


def test_6_depth_coherence():
    """Фазовая когерентность не деградирует к концу длинного контекста."""
    print("\n" + "=" * 60)
    print("TEST 6: Depth-Dependent Coherence")
    print("=" * 60)

    rc_calc = ResonanceConfidence()
    results = {}

    for ctx in CONTEXT_SIZES:
        if ctx < 2048:
            results[ctx] = {"passed": True, "note": "too short for quarter analysis"}
            print(f"  ctx={ctx:>6,}  |  too short for quarter analysis  [SKIP]")
            continue

        torch.manual_seed(42)
        model = build_model()
        warmup_model(model, ctx)

        positions = torch.arange(ctx, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(positions)

        phases = out["phases"]
        quarter = ctx // 4
        quarter_rcs = []
        quarter_names = ["Q1 (start)", "Q2", "Q3", "Q4 (end)"]

        for q in range(4):
            start = q * quarter
            end = start + quarter
            q_phases = phases[:, start:end, :]
            q_rc = rc_calc(q_phases).item()
            quarter_rcs.append(q_rc)

        ratio_q4_q1 = quarter_rcs[3] / quarter_rcs[0] if quarter_rcs[0] > 1e-8 else 0.0
        passed = ratio_q4_q1 > 0.85

        results[ctx] = {
            "quarter_rcs": {name: round(rc, 6) for name, rc in zip(quarter_names, quarter_rcs)},
            "ratio_q4_q1": round(ratio_q4_q1, 6),
            "passed": passed,
        }
        status = "PASS" if passed else "FAIL"
        rc_str = "  ".join(f"{name}={rc:.4f}" for name, rc in zip(quarter_names, quarter_rcs))
        print(f"  ctx={ctx:>6,}  |  {rc_str}")
        print(f"             |  Q4/Q1 = {ratio_q4_q1:.4f}  [{status}]")

    all_passed = all(r["passed"] for r in results.values())
    print(f"\n  Depth Coherence: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return {"test": "depth_coherence", "passed": all_passed, "details": results}


# ==================== ВИЗУАЛИЗАЦИЯ ====================


def plot_results(all_tests):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("WFR Long Context Stability Test — Experiment 03", fontsize=14, fontweight="bold")

    # 1. Windowed RC distribution (largest context)
    ax = axes[0, 0]
    t4 = next((t for t in all_tests if t["test"] == "windowed_rc"), None)
    if t4:
        largest_ctx = max(k for k in t4["details"].keys() if isinstance(k, int))
        rcs = t4["details"][largest_ctx].get("window_rcs", [])
        if rcs:
            ax.plot(rcs, linewidth=0.8, color="#3b82f6", alpha=0.8)
            ax.axhline(y=np.mean(rcs), color="#ef4444", linestyle="--", label=f"mean={np.mean(rcs):.4f}")
            ax.set_xlabel("Window index")
            ax.set_ylabel("RC")
            ax.set_title(f"Windowed RC (ctx={largest_ctx:,}, w={WINDOW_SIZE})")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 2. RC per quarter across contexts
    ax = axes[0, 1]
    t6 = next((t for t in all_tests if t["test"] == "depth_coherence"), None)
    if t6:
        quarter_labels = ["Q1", "Q2", "Q3", "Q4"]
        colors = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]
        for ctx_key, ctx_data in t6["details"].items():
            ctx_val = int(ctx_key) if isinstance(ctx_key, str) else ctx_key
            if "quarter_rcs" in ctx_data:
                vals = list(ctx_data["quarter_rcs"].values())
                ax.plot(quarter_labels, vals, "o-", label=f"ctx={ctx_val:,}", linewidth=2, markersize=8)
        ax.set_ylabel("RC")
        ax.set_title("RC by Quarter (depth coherence)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3. Cross-context standing wave similarity
    ax = axes[0, 2]
    t3 = next((t for t in all_tests if t["test"] == "cross_context_standing_wave"), None)
    if t3:
        ctxs = sorted(k for k in t3["details"].keys() if isinstance(k, int))
        sims = [t3["details"][c]["cosine_sim_vs_512"] for c in ctxs]
        ax.bar([f"{c:,}" for c in ctxs], sims, color="#8b5cf6", alpha=0.8)
        ax.axhline(y=0.95, color="#ef4444", linestyle="--", label="threshold=0.95")
        ax.set_ylabel("Cosine Similarity vs ctx=512")
        ax.set_title("Standing Wave Stability (prefix 0..511)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # 4. Spike density per chunk (largest context)
    ax = axes[1, 0]
    t5 = next((t for t in all_tests if t["test"] == "spike_distribution"), None)
    if t5:
        largest_ctx = max(k for k in t5["details"].keys() if isinstance(k, int))
        data = t5["details"][largest_ctx]
        if "per_layer_spike_rates" in data:
            rates = data["per_layer_spike_rates"]
            ax.bar(range(len(rates)), rates, color="#22c55e", alpha=0.8)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Spike Rate")
            ax.set_title(f"Per-Layer Spike Rate (ctx={largest_ctx:,})")
            ax.grid(True, alpha=0.3)

    # 5. Spike density ratio across contexts
    ax = axes[1, 1]
    if t5:
        ctxs = sorted(k for k in t5["details"].keys() if isinstance(k, int))
        ratios = []
        for c in ctxs:
            d = t5["details"][c]
            ratios.append(d.get("max_min_ratio", 1.0))
        ax.bar([f"{c:,}" for c in ctxs], ratios, color="#f59e0b", alpha=0.8)
        ax.axhline(y=3.0, color="#ef4444", linestyle="--", label="threshold=3.0")
        ax.set_ylabel("Max/Min Spike Density Ratio")
        ax.set_title("Spike Uniformity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6. Summary: pass/fail
    ax = axes[1, 2]
    test_names = [t["test"] for t in all_tests]
    test_passed = [t["passed"] for t in all_tests]
    colors = ["#22c55e" if p else "#ef4444" for p in test_passed]
    short_names = [n.replace("_", "\n") for n in test_names]
    ax.barh(short_names, [1] * len(test_names), color=colors, alpha=0.8)
    ax.set_xlim(0, 1.2)
    ax.set_title("Test Results Summary")
    for i, p in enumerate(test_passed):
        ax.text(1.05, i, "PASS" if p else "FAIL", va="center", fontweight="bold",
                color="#22c55e" if p else "#ef4444")
    ax.set_xticks([])

    plt.tight_layout()
    path = OUTPUT_DIR / "stability_test.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlots saved: {path}")


# ==================== MAIN ====================


def run_all_tests():
    start_time = time.time()
    all_results = []

    all_results.append(test_1_determinism())
    all_results.append(test_2_phase_encoding_stability())
    all_results.append(test_3_cross_context_standing_wave())
    all_results.append(test_4_windowed_rc())
    all_results.append(test_5_spike_distribution())
    all_results.append(test_6_depth_coherence())

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for t in all_results:
        status = "PASS" if t["passed"] else "FAIL"
        emoji = "+" if t["passed"] else "!"
        print(f"  [{emoji}] {t['test']:.<45} {status}")

    total_passed = sum(1 for t in all_results if t["passed"])
    total_tests = len(all_results)
    print(f"\n  Result: {total_passed}/{total_tests} tests passed")
    print(f"  Total time: {total_time:.1f}s")

    output = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "N/A",
        "config": BEST_CONFIG,
        "context_sizes": CONTEXT_SIZES,
        "homeostatic_warmup": HOMEOSTATIC_WARMUP,
        "window_size": WINDOW_SIZE,
        "total_time_sec": round(total_time, 2),
        "summary": {
            "total_passed": total_passed,
            "total_tests": total_tests,
            "all_passed": total_passed == total_tests,
        },
        "tests": [],
    }

    for t in all_results:
        serializable = {
            "test": t["test"],
            "passed": t["passed"],
            "details": {},
        }
        for k, v in t.get("details", {}).items():
            serializable["details"][str(k)] = v
        output["tests"].append(serializable)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = OUTPUT_DIR / f"stability_test_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved: {json_path}")

    plot_results(all_results)

    return all_results


if __name__ == "__main__":
    run_all_tests()
    print("\nLong Context Stability Test complete.")
