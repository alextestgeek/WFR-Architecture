"""
WFR Parameter Search Tool — 4 Layers Focus
Инструмент для честного поиска параметров для 4 фрактальных уровней.
Не подгоняем результат. Фиксируем всё как есть.

Цель: понять реальное поведение теории.
"""

import sys
import torch
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wfr.core import WFRNetwork


@dataclass
class LayerParams:
    frequency: float
    threshold: float


@dataclass
class ExperimentConfig:
    target_mode: str
    layers: List[LayerParams]
    decay: float = 2.0
    weight_scale: float = 0.1
    name: str = ""
    description: str = ""


class WFRParameterSearch:
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / "parameter_search"
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.best_configs = []

    def run_config(self, config: ExperimentConfig, num_layers: int = 4, seq_lengths=[512, 2048, 8192]) -> Dict:
        print(f"\n{'='*70}")
        print(f"Эксперимент: {config.name or config.target_mode} | Layers: {num_layers}")
        print(f"Описание: {config.description}")
        print(f"Target mode: {config.target_mode}")
        for i, lp in enumerate(config.layers[:num_layers]):
            print(f"  Layer {i}: freq={lp.frequency:.2f}, threshold={lp.threshold:.3f}")
        print('='*70)

        model = WFRNetwork(num_phases=16, num_fractal_levels=6, num_resonance_layers=num_layers)
        model.target_mode = config.target_mode

        # Применяем параметры
        for i, layer in enumerate(model.resonance_layers):
            if i < len(config.layers):
                layer.frequency.data = torch.tensor(config.layers[i].frequency)
                layer.threshold.data = torch.tensor(config.layers[i].threshold)
            layer.decay.data = torch.tensor(config.decay)
            if hasattr(layer, 'layer_idx'):
                layer.layer_idx = i

        result = {
            "config": {
                "name": config.name,
                "target_mode": config.target_mode,
                "layers": [{"freq": lp.frequency, "thresh": lp.threshold} for lp in config.layers],
                "description": config.description
            },
            "contexts": {},
            "summary": {}
        }

        total_active = 0
        for seq_len in seq_lengths:
            positions = torch.arange(seq_len).unsqueeze(0)
            with torch.no_grad():
                output = model(positions)
            
            layer_spikes = output["layer_spikes"]
            rc = output["resonance_confidence"].item()
            spike_rates = [spk.mean().item() for spk in layer_spikes]
            
            active_count = sum(1 for sr in spike_rates if sr > 0.01)
            total_active += active_count

            ctx_data = {
                "rc": round(rc, 4),
                "spike_rates": [round(sr*100, 1) for sr in spike_rates],
                "active_layers": active_count
            }
            result["contexts"][seq_len] = ctx_data

            print(f"  ctx={seq_len:5d} | RC={rc:.4f} | Active={active_count}/4 | "
                  f"Spikes={[f'{s:.1f}' for s in ctx_data['spike_rates']]}%")

        avg_active = total_active / len(seq_lengths)
        result["summary"] = {
            "avg_active_layers": round(avg_active, 2),
            "min_rc": min(ctx["rc"] for ctx in result["contexts"].values())
        }

        self.results.append(result)
        
        if avg_active >= 2.0 and result["summary"]["min_rc"] > 0.65:
            self.best_configs.append(result)
            print("  >>> ХОРОШАЯ КОНФИГУРАЦИЯ <<<")
        
        return result

    def save(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = self.output_dir / f"experiment_4layers_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "total_experiments": len(self.results),
            "good_configs": len(self.best_configs),
            "results": self.results
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nРезультаты сохранены: {path}")
        print(f"Хороших конфигураций найдено: {len(self.best_configs)}")
        return path


# ====================== ПРЕСЕТЫ ДЛЯ 4 СЛОЁВ ======================

def main():
    searcher = WFRParameterSearch()
    
    print("=== ПРОВЕРКА МАСШТАБИРУЕМОСТИ ТЕОРИИ ===\n")
    print("Тестируем 4, 6, 8, 10 и 12 слоёв\n")
    print("Цель: понять, сохраняется ли работоспособность при увеличении количества фрактальных уровней.\n")

    layer_counts = [4, 6, 8, 10, 12]
    
    base_layers = [
        LayerParams(1.0, 0.20),
        LayerParams(1.8, 0.25),
        LayerParams(3.0, 0.30),
        LayerParams(5.0, 0.40),
        LayerParams(7.0, 0.45),
        LayerParams(9.0, 0.50),
        LayerParams(12.0, 0.55),
        LayerParams(15.0, 0.60),
        LayerParams(18.0, 0.65),
        LayerParams(22.0, 0.70),
        LayerParams(26.0, 0.75),
        LayerParams(30.0, 0.80),
    ]
    
    for n_layers in layer_counts:
        print(f"\n--- Тестирование {n_layers} слоёв ---")
        
        config = ExperimentConfig(
            target_mode="frequency",
            layers=base_layers[:n_layers],
            name=f"{n_layers}-Layers-Freq",
            description=f"Честный тест на {n_layers} слоёв"
        )
        
        searcher.run_config(config, num_layers=n_layers)
    
    searcher.save()
    print("\nПроверка масштабируемости завершена.")
    print("Результаты сохранены в parameter_search/")


if __name__ == "__main__":
    main()
