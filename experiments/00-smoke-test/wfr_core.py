"""
WFR-Architecture Core Components
Test 0: Smoke Test — проверка жизнеспособности базовой архитектуры

Компоненты:
  1. WavePhaseEncoder (WPE)
  2. FractalResonanceLayer
  3. ResonanceTriggeredSpike
  4. ResonanceConfidence
"""

import torch
import torch.nn as nn
import math


class WavePhaseEncoder(nn.Module):
    """
    Wave Phase Encoder (WPE)
    
    Кодирует позиции через набор фаз с фрактальной иерархией.
    Память O(1) на позицию, независимо от длины контекста.
    """

    def __init__(self, num_phases: int = 16, num_fractal_levels: int = 6):
        super().__init__()
        self.num_phases = num_phases
        self.num_fractal_levels = num_fractal_levels

        self.base_freqs = nn.Parameter(
            2.0 ** torch.arange(num_phases, dtype=torch.float32),
            requires_grad=False,
        )

        self.alpha = nn.Parameter(torch.randn(num_fractal_levels) * 0.1)
        self.beta = nn.Parameter(torch.randn(num_fractal_levels) * 0.5 + 1.0)
        self.gamma = nn.Parameter(torch.randn(num_fractal_levels) * math.pi)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: [batch, seq_len] — целые числа (позиции)
        returns:   [batch, seq_len, num_phases] — фазы в [0, 2π)
        """
        i = positions.unsqueeze(-1).float()
        f = self.base_freqs.unsqueeze(0).unsqueeze(0)

        base_phase = 2 * math.pi * f * i

        fractal_term = torch.zeros_like(base_phase)
        for l in range(self.num_fractal_levels):
            log_pos = torch.log2(i + 2)
            fractal_term += self.alpha[l] * torch.sin(
                2 * math.pi * self.beta[l] * log_pos + self.gamma[l]
            )

        phases = (base_phase + fractal_term) % (2 * math.pi)
        return phases


class FractalResonanceLayer(nn.Module):
    """
    Один уровень фрактального резонатора.
    
    Принимает фазы, вычисляет резонансную функцию,
    возвращает амплитуды резонанса.
    """

    def __init__(self, num_phases: int = 16, frequency: float = 1.0, threshold: float = 0.3):
        super().__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency))
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
        self.decay = nn.Parameter(torch.tensor(2.0))
        self.weights = nn.Parameter(torch.randn(num_phases) * 0.1)

    def resonance_function(self, z: torch.Tensor) -> torch.Tensor:
        """φ(z) = sin(2π·f·z) · exp(-α·|z - θ|) при |z| > θ"""
        amplitude = torch.sin(2 * math.pi * self.frequency * z)
        envelope = torch.exp(-self.decay * torch.abs(z - self.threshold))
        mask = torch.abs(z) > self.threshold
        return amplitude * envelope * mask.float()

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """
        phases: [batch, seq_len, num_phases]
        returns: [batch, seq_len] — амплитуда резонанса на этом уровне
        """
        weighted = phases * self.weights.unsqueeze(0).unsqueeze(0)
        combined = weighted.sum(dim=-1)
        resonance = self.resonance_function(combined)
        return resonance


class ResonanceTriggeredSpike(nn.Module):
    """
    Event-driven спайковый механизм.
    
    Спайк генерируется только при конструктивной интерференции
    выше порога.
    """

    def __init__(self, spike_threshold: float = 0.5):
        super().__init__()
        self.spike_threshold = nn.Parameter(
            torch.tensor(spike_threshold), requires_grad=False
        )

    def forward(self, resonance: torch.Tensor) -> torch.Tensor:
        """
        resonance: [batch, seq_len] — амплитуды резонанса
        returns:   [batch, seq_len] — бинарные спайки (0 или 1)
        """
        spikes = (torch.abs(resonance) > self.spike_threshold).float()
        return spikes


class ResonanceConfidence(nn.Module):
    """
    Вычисляет метрику устойчивости резонансного паттерна (RC).
    
    RC = средняя когерентность фаз по последовательности.
    Высокий RC → устойчивая стоячая волна.
    Низкий RC → шум или нестабильный паттерн.
    """

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """
        phases: [batch, seq_len, num_phases]
        returns: скаляр RC в [0, 1]
        """
        complex_phases = torch.exp(1j * phases)
        mean_vector = complex_phases.mean(dim=1)
        coherence = torch.abs(mean_vector).mean()
        return coherence


class WFRNetwork(nn.Module):
    """
    Полная WFR-сеть для Smoke Test.
    
    WPE → Fractal Resonance Layers → Spiking → Standing Wave Output
    """

    def __init__(
        self,
        num_phases: int = 16,
        num_fractal_levels: int = 6,
        num_resonance_layers: int = 4,
    ):
        super().__init__()
        self.encoder = WavePhaseEncoder(num_phases, num_fractal_levels)

        self.resonance_layers = nn.ModuleList([
            FractalResonanceLayer(
                num_phases=num_phases,
                frequency=1.0 + i * 0.5,
                threshold=0.2 + i * 0.05,
            )
            for i in range(num_resonance_layers)
        ])

        self.spike_trigger = ResonanceTriggeredSpike(spike_threshold=0.4)
        self.confidence = ResonanceConfidence()

    def forward(self, positions: torch.Tensor):
        """
        positions: [batch, seq_len]
        returns: dict с полной диагностикой
        """
        phases = self.encoder(positions)

        layer_resonances = []
        layer_spikes = []

        for layer in self.resonance_layers:
            resonance = layer(phases)
            spikes = self.spike_trigger(resonance)
            layer_resonances.append(resonance)
            layer_spikes.append(spikes)

        rc = self.confidence(phases)

        standing_wave = torch.stack(layer_resonances, dim=0).mean(dim=0)

        return {
            "phases": phases,
            "layer_resonances": layer_resonances,
            "layer_spikes": layer_spikes,
            "resonance_confidence": rc,
            "standing_wave": standing_wave,
        }
