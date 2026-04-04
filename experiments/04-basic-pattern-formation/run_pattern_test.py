"""
WFR Basic Pattern Formation — Experiment 04
============================================

Проверка теории (docs/03-theory.md, раздел 5): разные структуры входа формируют
различимые устойчивые паттерны (стоячие волны, RC, спайковый профиль).

Без обучения: одна и та же сеть, синтетические последовательности позиций
одинаковой длины, замороженный homeostatic после warmup.

Этапы:
  A — различимость классов паттернов (pairwise cosine сигнатур)
  B — повторяемость: тот же паттерн → та же сигнатура
  C (опц.) — лёгкий шум по фазам после WPE (устойчивость)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from wfr.core import ResonanceConfidence, WFRNetwork, surrogate_spike

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEQ_LEN = 4096
HOMEOSTATIC_WARMUP = 50
PHASE_NOISE_STD = 0.02

BEST_CONFIG = {
    "target_mode": "frequency",
    "frequencies": [1.0, 1.8, 3.0, 5.0],
    "thresholds": [0.20, 0.25, 0.30, 0.40],
    "name": "Freq-Balanced",
}


# --- Паттерны позиций [1, L], int64 (как в остальных тестах) ---


def pat_linear(L: int, device):
    return torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)


def pat_reverse(L: int, device):
    return torch.arange(L - 1, -1, -1, device=device, dtype=torch.long).unsqueeze(0)


def pat_mod_small_period(L: int, device, period: int = 64):
    """Повторяющийся блок малых индексов — сильно отличается от линейного прохода."""
    v = torch.arange(L, device=device) % period
    return v.long().unsqueeze(0)


def pat_stride_mod(L: int, device, k: int = 7):
    """«Прогулка» по кольцу индексов: (i * k) % L."""
    Lm = max(L, 1)
    v = (torch.arange(L, device=device, dtype=torch.long) * k) % Lm
    return v.unsqueeze(0)


def pat_shuffle(L: int, device, seed: int = 12345):
    """Перестановка значений 0..L-1 вдоль последовательности (тот же multiset, другой порядок)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(L, generator=g, device=device)
    base = torch.arange(L, device=device, dtype=torch.long)
    return base[perm].unsqueeze(0)


def pat_two_blocks(L: int, device):
    """Два одинаковых полублока 0..L/2-1 (структурное повторение)."""
    half = L // 2
    a = torch.arange(half, device=device, dtype=torch.long)
    v = torch.cat([a, a.clone()], dim=0)
    if v.numel() < L:
        v = torch.cat([v, torch.arange(v.numel(), L, device=device, dtype=torch.long)], dim=0)
    return v[:L].unsqueeze(0)


PATTERNS = {
    "linear": pat_linear,
    "reverse": pat_reverse,
    "mod64": lambda L, d: pat_mod_small_period(L, d, 64),
    "stride7_mod_L": pat_stride_mod,
    "shuffle_seed12345": pat_shuffle,
    "two_blocks": pat_two_blocks,
}


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


def warmup_linear(model, L: int):
    pos = torch.arange(L, device=DEVICE, dtype=torch.long).unsqueeze(0)
    for _ in range(HOMEOSTATIC_WARMUP):
        with torch.no_grad():
            model(pos)


def freeze_homeostatic(model):
    for layer in model.resonance_layers:
        layer.homeostatic_enabled = False


@torch.no_grad()
def forward_with_optional_phase_noise(model, positions: torch.Tensor, phase_noise: float = 0.0):
    phases = model.encoder(positions)
    if phase_noise > 0:
        phases = phases + phase_noise * torch.randn_like(phases)
        phases = phases % (2 * np.pi)

    layer_resonances = []
    layer_spikes = []
    for layer in model.resonance_layers:
        resonance = layer(phases, target_mode=model.target_mode)
        spikes = surrogate_spike(
            torch.abs(resonance), layer.spike_threshold, layer_idx=layer.layer_idx
        )
        layer_resonances.append(resonance)
        layer_spikes.append(spikes)

    rc_calc = ResonanceConfidence()
    rc = rc_calc(phases)
    standing_wave = torch.stack(layer_resonances, dim=0).mean(dim=0)
    return {
        "phases": phases,
        "layer_resonances": layer_resonances,
        "layer_spikes": layer_spikes,
        "resonance_confidence": rc,
        "standing_wave": standing_wave,
    }


def signature_vector(out: dict) -> np.ndarray:
    """Сигнатура паттерна: профиль стоячей волны по четвертям + слои + RC.

    Глобальные mean/std недостаточны — разные порядка позиций дают одинаковые
    агрегаты; четверти ловят структуру вдоль последовательности (теория, раздел 5).
    """
    sw = out["standing_wave"].float().squeeze(0)
    L = sw.numel()
    q = max(L // 4, 1)
    quarters = []
    for k in range(4):
        sl = sw[k * q : (k + 1) * q] if k < 3 else sw[k * q :]
        quarters.append(sl.mean().unsqueeze(0))
    sw_std = sw.std().unsqueeze(0)
    rates = torch.stack([s.mean() for s in out["layer_spikes"]])
    rc = out["resonance_confidence"].flatten().float()
    vec = torch.cat(quarters + [sw_std, rates, rc], dim=0)
    x = vec.cpu().numpy().astype(np.float64)
    n = np.linalg.norm(x)
    if n > 1e-12:
        x = x / n
    return x


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_phase_a_b(model, L: int):
    """Этапы A и B: различимость и повторяемость."""
    names = list(PATTERNS.keys())
    sigs = {}
    raw_out = {}

    for name in names:
        positions = PATTERNS[name](L, DEVICE)
        with torch.no_grad():
            out = model(positions)
        raw_out[name] = {
            "rc": round(float(out["resonance_confidence"].item()), 6),
            "spike_rates": [round(float(s.mean().item()), 6) for s in out["layer_spikes"]],
        }
        sigs[name] = signature_vector(out)

    n = len(names)
    cos_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = cosine_np(sigs[names[i]], sigs[names[j]])
            cos_mat[i, j] = cos_mat[j, i] = c

    off = []
    for i in range(n):
        for j in range(i + 1, n):
            off.append(cos_mat[i, j])
    max_off_cos = max(off) if off else 0.0
    min_off_cos = min(off) if off else 0.0
    mean_off_cos = float(np.mean(off)) if off else 0.0

    # Повторяемость (B): linear дважды
    p0 = PATTERNS["linear"](L, DEVICE)
    with torch.no_grad():
        o1 = model(p0)
        o2 = model(p0)
    s1 = signature_vector(o1)
    s2 = signature_vector(o2)
    repeat_cos = cosine_np(s1, s2)

    return {
        "names": names,
        "signatures_dim": len(sigs[names[0]]),
        "pairwise_cosine": {names[i]: {names[j]: float(cos_mat[i, j]) for j in range(n)} for i in range(n)},
        "off_diagonal_max": round(max_off_cos, 8),
        "off_diagonal_min": round(min_off_cos, 8),
        "off_diagonal_mean": round(mean_off_cos, 8),
        "repeatability_linear_cos": round(repeat_cos, 10),
        "per_pattern": raw_out,
    }


def run_phase_c(model, L: int):
    """Лёгкий шум по фазам после WPE — два уровня шума, паттерн linear."""
    detail = {}
    pos = PATTERNS["linear"](L, DEVICE)
    full_sigs = {}
    for noise in (0.0, PHASE_NOISE_STD, PHASE_NOISE_STD * 2.5):
        torch.manual_seed(42)
        if noise == 0.0:
            with torch.no_grad():
                out = model(pos)
        else:
            with torch.no_grad():
                out = forward_with_optional_phase_noise(model, pos, phase_noise=noise)
        vec = signature_vector(out)
        full_sigs[noise] = vec
        detail[str(noise)] = {
            "rc": round(float(out["resonance_confidence"].item()), 6),
            "signature_head": vec.tolist()[:8],
        }
    noise_cos = cosine_np(full_sigs[0.0], full_sigs[PHASE_NOISE_STD])
    return {
        "phase_noise_std": PHASE_NOISE_STD,
        "cosine_clean_vs_noisy": round(noise_cos, 8),
        "detail": detail,
    }


def plot_cosine_matrix(names, cos_mat, path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cos_mat, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Pairwise cosine similarity of pattern signatures\n(standing-wave stats + spike rates + RC)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    print("Basic Pattern Formation — Experiment 04")
    print(f"Device: {DEVICE} | seq_len={SEQ_LEN}")
    print("=" * 72)

    torch.manual_seed(42)
    model = build_model()
    warmup_linear(model, SEQ_LEN)
    freeze_homeostatic(model)
    print(f"Warmup: linear L={SEQ_LEN}, passes={HOMEOSTATIC_WARMUP}, homeostatic frozen.")

    phase_ab = run_phase_a_b(model, SEQ_LEN)
    phase_c = run_phase_c(model, SEQ_LEN)

    max_off = phase_ab["off_diagonal_max"]
    repeat_ok = phase_ab["repeatability_linear_cos"] > 1.0 - 1e-5

    # Пары паттернов не должны быть идентичны по сигнатуре (численный порог)
    distinguish_ok = max_off < 0.99999
    strong_ok = max_off < 0.995

    noise_cos = phase_c["cosine_clean_vs_noisy"]
    noise_ok = noise_cos > 0.85

    print("\n--- Phase A: Distinguishability ---")
    print(f"  Pairwise cos: min={phase_ab['off_diagonal_min']:.6f}  max={max_off:.6f}  mean={phase_ab['off_diagonal_mean']:.6f}")
    print(f"  PASS (max off-diag cos < 0.99999): {distinguish_ok}")
    print(f"  STRONG (max off-diag cos < 0.995): {strong_ok}")

    print("\n--- Phase B: Repeatability (linear x2) ---")
    print(f"  Cosine same pattern: {phase_ab['repeatability_linear_cos']}")
    print(f"  PASS: {repeat_ok}")

    print("\n--- Phase C: Phase noise (linear) ---")
    print(f"  Cos(clean, noisy std={PHASE_NOISE_STD}): {noise_cos}")
    print(f"  PASS (cos > 0.85): {noise_ok}")

    out = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "seq_len": SEQ_LEN,
        "config": BEST_CONFIG,
        "warmup_passes": HOMEOSTATIC_WARMUP,
        "phase_ab": phase_ab,
        "phase_c": phase_c,
        "criteria": {
            "distinguishability_pass": bool(distinguish_ok),
            "distinguishability_strong": bool(strong_ok),
            "repeatability_pass": bool(repeat_ok),
            "noise_robustness_pass": bool(noise_ok),
        },
        "verdict": (
            "Подтверждено"
            if distinguish_ok and repeat_ok
            else ("Частично" if distinguish_ok or repeat_ok else "Требует доработки")
        ),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = OUTPUT_DIR / f"pattern_test_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    names = phase_ab["names"]
    n = len(names)
    cos_mat = np.eye(n)
    for i in range(n):
        for j in range(n):
            cos_mat[i, j] = phase_ab["pairwise_cosine"][names[i]][names[j]]

    plot_cosine_matrix(names, cos_mat, OUTPUT_DIR / "pattern_cosine_matrix.png")

    print(f"\nJSON: {json_path}")
    print(f"Plot: {OUTPUT_DIR / 'pattern_cosine_matrix.png'}")
    print(f"Вердикт: {out['verdict']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
