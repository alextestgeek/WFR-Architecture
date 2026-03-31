# WFR-Architecture

**Wave-Fractal-Resonant Architecture** — a novel AI architecture that replaces matrix multiplications with wave physics, resonance, and fractal self-organization.

## Core Idea

Instead of computing through linear transformations, information **resonates** within the system, forming stable standing wave patterns in a multi-dimensional phase space.

## Architecture Components

### Wave Phase Encoder (WPE)

Encodes each token position as a set of phases with fractal hierarchy:

$$
\phi_i^{(m)} = \left( 2\pi f_m i + \sum_{l=1}^L \alpha_l \sin(2\pi \beta_l \log_2(i+2) + \gamma_l) \right) \bmod 2\pi
$$

- O(1) memory per token (confirmed experimentally up to 100M tokens)
- Hierarchical distance encoding via logarithmic fractal terms

### Fractal Resonance Layers

Multi-level fractal structure where each level handles its own scale of patterns:

$$
\phi_l(z) = \sin(2\pi f_l z) \cdot e^{-\alpha |z-\theta_l|} \quad \text{when } |z| > \theta_l
$$

### Resonance-Triggered Spiking (Event-Driven)

Spikes fire only on constructive interference — the system is inherently event-driven:

$$
R = \Re\left(\sum_j w_j \cdot e^{i(\phi_j - \phi_{\text{target}})}\right), \quad \text{spike} \iff |R| > \theta
$$

### Stability Mechanisms (v2.0, March 2026)

Three mechanisms added to scale reliably to 100M+ tokens:

1. **Phase-Locking (WPE-L)** — global phase synchronization every 4 frequencies. Subtracts the circular mean phase of each group, analogous to θ-rhythm phase reset in biological oscillators:

$$
\phi^{(l)}_{i,m} \leftarrow \phi^{(l)}_{i,m} - \arg\!\left( \frac{1}{M} \sum_{j=1}^{M} e^{i \phi^{(l)}_{i,j}} \right)
$$

2. **Homeostatic Spike Threshold** — adaptive threshold keeps spike rate near a 10% target via negative feedback:

$$
\theta_l(t+1) = \theta_l(t) + \eta \cdot (r_{\text{real}}(t) - r_{\text{target}})
$$

3. **Multi-scale Surrogate Gradient** — differentiable spiking with level-dependent scaling (γ=0.92 on key levels, 0.98 otherwise) to prevent vanishing gradients across fractal depth.

### Learning (Planned): Resonant Field Plasticity (RFP)

- Two modes: real-time (online) and accelerated (offline)
- Loss function balances task performance, resonance confidence (RC), and energy cost
- Learning adjusts frequencies and phases, not classical weight matrices

## Experimental Results (NVIDIA A100 80GB)

Tested up to **100,000,000 tokens** in a single forward pass.

| Context       | VRAM     | Mem/token | Time     | RC (v1.0) | RC (v2.0) |
|---------------|----------|-----------|----------|-----------|-----------|
| 512           | 0.3 MB   | 492 bytes | 0.14 s   | 0.93      | **0.97**  |
| 131,072       | 68.8 MB  | 492 bytes | 0.004 s  | 0.46      | **0.71**  |
| 1,048,576     | 550 MB   | 492 bytes | 0.013 s  | 0.28      | **0.61**  |
| 16,777,216    | 8.4 GB   | 492 bytes | 0.092 s  | 0.05      | **0.48**  |
| **100,000,000** | **54.1 GB** | **492 bytes** | **2.71 s** | **0.011** | **0.464** |

### What is confirmed

- **O(1) memory per token** — constant 492 bytes across the entire range (512 to 100M)
- **Phase-Locking** improves coherence by **×42** at 100M tokens (RC: 0.011 → 0.464)
- Architecture is **viable** — produces standing waves and event-driven behavior

### Layer Scaling Test (v2.1 — Homeostatic Bugfix)

Critical bug found: sign inversion in homeostatic regulation created **positive feedback** (silent layers → threshold increases → even more silent). After fixing one line, all 32 layers become active:

| Layers | ctx=512 | ctx=8,192 | ctx=131,072 |
|--------|---------|-----------|-------------|
| 4      | 100%    | 100%      | 100%        |
| 8      | 100%    | 100%      | 100%        |
| 16     | 88–94%  | **100%**  | **100%**    |
| 24     | 92–100% | **100%**  | **100%**    |
| **32** | 91–97%  | **100%**  | **100%**    |

RC is stable across depths: ~0.97 (512 tokens), ~0.846 (8K), ~0.712 (131K) — independent of layer count.

### What is not confirmed

- **Time complexity** remains ~O(n), not sub-linear as theorized
- **Learning (RFP)** — not yet implemented; only forward pass tested
- **Comparison with existing architectures** — not yet conducted

Full results: [`docs/09-memory-complexity-test-plan.md`](docs/09-memory-complexity-test-plan.md)

## Project Structure

```
docs/               Theory and documentation
experiments/        Test code and results
  00-smoke-test/      Smoke Test + core implementation (wfr_core.py)
  03-memory-test/     Memory & Complexity Test (up to 100M tokens)
  04-layer-scaling/   Layer Scaling Test (up to 32 layers)
tools/              Utilities (interactive visualizer)
```

## Documentation

1. [`docs/00-overview.md`](docs/00-overview.md) — Architecture overview
2. [`docs/02-architecture.md`](docs/02-architecture.md) — Components and data flow
3. [`docs/03-theory.md`](docs/03-theory.md) — Mathematical foundations + v2.0 stability mechanisms
4. [`docs/09-memory-complexity-test-plan.md`](docs/09-memory-complexity-test-plan.md) — Test results with v1.0 vs v2.0 comparison

## License

**AGPL-3.0** — see [LICENSE](LICENSE).

All commits are GPG-signed for verified authorship.

---

Development actively uses neural network models (LLMs) as tools for research, code generation, analysis, and documentation.

**Principle:** Experimental honesty matters more than confirming a hypothesis.
