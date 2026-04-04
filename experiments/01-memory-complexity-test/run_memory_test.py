"""
WFR Memory & Complexity Test
Проверка утверждений теории о O(1)/O(log n) памяти и сложности.

Мы честно измеряем:
- Память на токен при росте контекста
- Время forward pass
- Эмпирическую сложность
"""

import torch
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

import sys
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from wfr.core import WFRNetwork

# ==================== КОНФИГУРАЦИЯ ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Поэтапное тестирование до 100 миллионов токенов
# Начинаем с маленьких, затем резко увеличиваем, чтобы увидеть поведение
CONTEXT_SIZES = [
    512, 
    2048, 
    8192, 
    16384, 
    32768, 
    65536, 
    131072,      # 131K
    524288,      # 0.5M
    1048576,     # 1M
    4194304,     # 4M
    16777216,    # 16M
    33554432,    # 32M
    67108864,    # 64M
    100000000    # 100M (цель)
]
BEST_CONFIG = {
    "target_mode": "frequency",
    "frequencies": [1.0, 1.8, 3.0, 5.0],
    "thresholds": [0.20, 0.25, 0.30, 0.40],
    "name": "Freq-Balanced"
}

print(f"Memory & Complexity Test")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"CUDA: {torch.cuda.get_device_name(0)} (capability {torch.cuda.get_device_capability(0)})")
print(f"Testing contexts: {CONTEXT_SIZES}")
print("="*80)


def measure_memory_and_time():
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": BEST_CONFIG,
        "contexts": {},
        "summary": {}
    }
    
    model = WFRNetwork(
        num_phases=16,
        num_fractal_levels=6,
        num_resonance_layers=4,
    ).to(DEVICE)
    model.target_mode = BEST_CONFIG["target_mode"]
    
    # Применяем параметры
    for i, layer in enumerate(model.resonance_layers):
        if i < len(BEST_CONFIG["frequencies"]):
            layer.frequency.data = torch.tensor(BEST_CONFIG["frequencies"][i], device=DEVICE)
            layer.spike_threshold.data = torch.tensor(BEST_CONFIG["thresholds"][i], device=DEVICE)
    
    for seq_len in CONTEXT_SIZES:
        print(f"\nTesting context = {seq_len:,}")
        
        positions = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        
        # Замер памяти и времени
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
            print(f"  VRAM before: {mem_before / (1024**2):.1f} MB")
        else:
            mem_before = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            output = model(positions)
        if seq_len == CONTEXT_SIZES[0]:
            assert output["phases"].device == positions.device
            assert output["phases"].device.type == DEVICE.type, (
                f"expected tensors on {DEVICE.type}, got {output['phases'].device}"
            )
        
        elapsed = time.time() - start_time
        
        if DEVICE.type == 'cuda':
            mem_after = torch.cuda.max_memory_allocated()
            mem_used = mem_after - mem_before
            mem_per_token = mem_used / seq_len if seq_len > 0 else 0
            print(f"  VRAM after:  {mem_after / (1024**2):.1f} MB (+{mem_used / (1024**2):.1f} MB)")
        else:
            mem_used = 0
            mem_per_token = 0
        
        rc = output["resonance_confidence"].item()
        active_layers = sum(1 for spk in output["layer_spikes"] if spk.mean().item() > 0.01)
        
        ctx_result = {
            "context": seq_len,
            "rc": round(rc, 4),
            "time_sec": round(elapsed, 4),
            "memory_bytes": int(mem_used),
            "memory_per_token_bytes": round(mem_per_token, 2),
            "active_layers": active_layers,
            "total_layers": 4
        }
        
        results["contexts"][seq_len] = ctx_result
        
        print(f"  RC: {rc:.4f} | Time: {elapsed:.4f}s | "
              f"Mem/token: {mem_per_token/1024:.1f} KB | Active: {active_layers}/4")
    
    # Сохраняем результаты
    save_results(results)
    plot_results(results)
    
    return results


def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = OUTPUT_DIR / f"memory_test_{timestamp}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nРезультаты сохранены: {path}")


def plot_results(results):
    contexts = list(results["contexts"].keys())
    mem_per_token = [results["contexts"][c]["memory_per_token_bytes"]/1024 for c in contexts]
    times = [results["contexts"][c]["time_sec"] for c in contexts]
    
    # Добавляем логарифмическую шкалу для больших значений
    import numpy as np
    log_contexts = [np.log10(c) for c in contexts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(contexts, mem_per_token, 'o-', label='Memory per token (KB)')
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Memory per Token (KB)')
    ax1.set_title('Memory Scaling (should be flat = O(1))')
    ax1.set_xscale('log')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(contexts, times, 'o-', label='Time (seconds)', color='orange')
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time Scaling')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "memory_complexity.png", dpi=150)
    plt.close()
    print(f"Графики сохранены: {OUTPUT_DIR}/memory_complexity.png")


if __name__ == "__main__":
    measure_memory_and_time()
    print("\nТест на память и сложность завершён.")
