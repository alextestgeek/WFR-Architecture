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


class PhaseInterference(nn.Module):
    """
    Phase Interference Calculator — точная реализация по теории.
    
    Вычисляет меру резонанса R = Re(e^{i(φ_j - φ_target)}) = cos(Δφ)
    Используется для Resonance-Triggered Spiking.
    """

    def __init__(self):
        super().__init__()

    def resonance_measure(self, phases: torch.Tensor, target_phase: torch.Tensor = None) -> torch.Tensor:
        """
        Вычисляет конструктивную интерференцию между фазами.
        R(φ_j, φ_target) = cos(φ_j - φ_target)
        
        phases: [batch, seq_len, num_phases]
        target_phase: optional [batch, seq_len, 1]
        """
        if target_phase is None:
            target_phase = phases.mean(dim=-1, keepdim=True)
        
        delta = phases - target_phase
        resonance = torch.cos(delta)  # R = cos(Δφ)
        return resonance

    def complex_interference(self, phases: torch.Tensor, weights: torch.Tensor, target_phase: torch.Tensor = None) -> torch.Tensor:
        """
        Комплексная версия по теории:
        R = Re(∑_j w_j · e^{i(φ_j - φ_target)})
        """
        if target_phase is None:
            target_phase = phases.mean(dim=-1, keepdim=True)
        
        # e^{i(φ_j - φ_target)} = e^{iφ_j} * e^{-iφ_target}
        complex_phases = torch.exp(1j * phases)
        complex_target = torch.exp(-1j * target_phase)
        aligned = complex_phases * complex_target
        weighted = aligned * weights.unsqueeze(0).unsqueeze(0)
        interference = weighted.sum(dim=-1)
        return torch.real(interference)


class TheoreticalResonanceLayer(nn.Module):
    """
    Теоретически корректный резонансный слой.
    
    Использует PhaseInterference + resonance_function по формуле из теории.
    """

    def __init__(self, num_phases: int = 16, frequency: float = 1.0, threshold: float = 0.3, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.frequency = nn.Parameter(torch.tensor(frequency))
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
        self.decay = nn.Parameter(torch.tensor(2.0))
        self.weights = nn.Parameter(torch.randn(num_phases) * 0.1)
        self.interference = PhaseInterference()
        
        print(f"  Layer {layer_idx}: freq={frequency:.2f}, threshold={threshold:.3f}")

    def resonance_function(self, z: torch.Tensor) -> torch.Tensor:
        """φ(z) = sin(2π·f·z) · exp(-α·|z - θ|) при |z| > θ"""
        amplitude = torch.sin(2 * math.pi * self.frequency * z)
        envelope = torch.exp(-self.decay * torch.abs(z - self.threshold))
        mask = torch.abs(z) > self.threshold
        return amplitude * envelope * mask.float()

    def forward(self, phases: torch.Tensor, target_mode: str = "mean") -> torch.Tensor:
        """
        phases: [batch, seq_len, num_phases]
        target_mode: "mean", "frequency", "base"
        returns: [batch, seq_len] — resonance amplitude
        """
        batch, seq_len, _ = phases.shape
        
        if target_mode == "mean":
            # Вариант A: средняя фаза по всем компонентам
            target_phase = phases.mean(dim=-1, keepdim=True)
        elif target_mode == "frequency":
            # Вариант B: собственная частота слоя
            target_phase = torch.full((batch, seq_len, 1), self.frequency.item() * 0.1, 
                                    device=phases.device, dtype=phases.dtype)
        elif target_mode == "base":
            # Вариант C: первая компонента фаз (базовая частота)
            target_phase = phases[:, :, :1]
        else:
            target_phase = phases.mean(dim=-1, keepdim=True)
        
        # 1. Фазовая интерференция по теории
        interference = self.interference.complex_interference(
            phases, self.weights, target_phase
        )
        
        # 2. Применяем резонансную функцию
        resonance = self.resonance_function(interference)
        
        return resonance


class WFRNetwork(nn.Module):
    """
    Полная WFR-сеть — ТЕОРЕТИЧЕСКИ КОРРЕКТНАЯ версия.
    
    Поток: WPE → PhaseInterference → TheoreticalResonanceLayer → Standing Wave
    Полностью соответствует формулам из 02-architecture.md и 03-theory.md.
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
            TheoreticalResonanceLayer(
                num_phases=num_phases,
                frequency=1.0 + i * 0.5,
                threshold=0.2 + i * 0.05,
                layer_idx=i
            )
            for i in range(num_resonance_layers)
        ])
        self.target_mode = "frequency"  # лучший вариант по предыдущим тестам

        self.confidence = ResonanceConfidence()

    def forward(self, positions: torch.Tensor):
        """
        positions: [batch, seq_len]
        returns: dict с полной диагностикой
        
        Теперь используем теоретически корректный путь:
        WPE → TheoreticalResonanceLayer (с PhaseInterference внутри)
        """
        phases = self.encoder(positions)

        layer_resonances = []
        layer_spikes = []
        layer_interferences = []

        for layer in self.resonance_layers:
            resonance = layer(phases, target_mode=self.target_mode)
            
            # Теоретический спайкинг: |R| > threshold
            spikes = (torch.abs(resonance) > 0.4).float()
            
            layer_resonances.append(resonance)
            layer_spikes.append(spikes)
            layer_interferences.append(resonance)

        rc = self.confidence(phases)
        standing_wave = torch.stack(layer_resonances, dim=0).mean(dim=0)

        return {
            "phases": phases,
            "layer_resonances": layer_resonances,
            "layer_spikes": layer_spikes,
            "resonance_confidence": rc,
            "standing_wave": standing_wave,
            "interferences": layer_interferences,
        }
