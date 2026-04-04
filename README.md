# WFR-Architecture

**Wave-Fractal-Resonant Architecture** — a novel AI architecture that replaces matrix multiplications with wave physics, resonance, and fractal self-organization.

## How It Works — Plain Language

Modern neural networks (GPT, Claude, etc.) are built on **transformers** that multiply huge matrices of numbers. The longer the input text, the more memory and compute they need — often quadratically.

WFR takes a different path: instead of matrix multiplication, information is encoded as **wave phases** — like radio stations on different frequencies. Processing happens through **resonance**: waves reinforce or cancel each other naturally, without multiplying tables of numbers.

### The Pipeline

1. **Wave Phase Encoder** — each word gets a unique "melody" based on its position. A fractal structure (nested scales, like a Matryoshka doll) lets the system tell apart "nearby" and "far away" without memorizing the full text. Memory per word is **fixed at 492 bytes** — whether the text has 500 words or 100 million.

2. **Fractal Resonance Layers** — a stack of "tuning forks," each tuned to its own frequency. Low layers catch broad patterns (topic), high layers catch fine ones (specific words). A layer resonates when its frequency matches the incoming signal; otherwise it stays silent.

3. **Spike Detector** — when waves from several layers align (constructive interference), a **spike** fires. No alignment — no computation. The system is event-driven: it only reacts to meaningful coincidences.

4. **Self-Tuning (v2.0–v2.1)**:
   - **Phase-Locking** — periodic re-synchronization, like a conductor keeping an orchestra in time. Without it, long texts lose coherence.
   - **Homeostatic Regulation** — automatic sensitivity adjustment per layer. Silent layer → lower threshold. Overactive → raise threshold. Works like a thermostat.
   - **Surrogate Gradient** — makes the discrete spike mechanism smooth enough for gradient-based training.

### What's proven

- **O(1) memory** — 492 bytes/token from 512 to 100M tokens (a transformer would need terabytes at 100M).
- **32 active layers** — after fixing a sign bug in homeostatic regulation, all layers stay alive.
- **Phase-Locking** boosts stability **×42** on long contexts.
- **Pattern-order signatures (Phase 0, no training)** — six synthetic position orders at fixed length yield **non-collapsed** aggregated signatures (standing-wave quarters + spikes + RC; Experiment 04). Strong pairwise separation everywhere is **not** claimed.

### What's not proven yet

- **Learning at scale** — toy next-token training and **RFP v0** (Adam + composite loss + plasticity step on frequencies / phase_bias / decay) are implemented (`wfr_lm.py`, `wfr_rfp.py`, [`experiments/06-rfp-v0/`](experiments/06-rfp-v0/), [`docs/11-rfp-v0-spec.md`](docs/11-rfp-v0-spec.md)); strong CE gains vs baseline and long-run criteria are still **Phase 2** work.
- **Speed** — still ~O(n) in practice, not sub-linear.
- **Head-to-head with transformers** — **partially**: char WikiText-2, fair parity vs a minimal **matched** causal Transformer (Experiment 09); the **readout / head width** is the main CE lever; synthesis [`docs/13-project-status-snapshot.md`](docs/13-project-status-snapshot.md) (§7), matrix [`docs/14-core-readiness-and-breakthrough-matrix.md`](docs/14-core-readiness-and-breakthrough-matrix.md).

### Training: synthetic evidence and next step (April 2026)

**Established on synthetic / toy runs** (Exp 05–07, protocol tests on GPU): the stack **can train** — gradients flow; Adam + composite loss + RFP (and related precursors) run without collapsing. That supports **Phase 1** engineering closure in [`docs/10-phase-1-plan.md`](docs/10-phase-1-plan.md) (sanity, explicit loss, reproducible scripts).

**Not established there:** a proof of large CE improvement or RFP dominance — small vocab and toy sampling often keep val CE near \(\ln V\); that is a **weak task signal**, not a verdict on the architecture at scale.

**Next step:** a small **real** text corpus for calibration — e.g. **WikiText-2** raw via [`data/hf/download_wikitext2.py`](data/hf/download_wikitext2.py) ([`data/hf/README.md`](data/hf/README.md)). That is **Phase 2+** exploration, not standard LM leaderboards.

> **One-sentence analogy:** WFR replaces "a calculator with giant spreadsheets" (transformer) with "a room full of tuning forks" where information travels through wave resonance, not number crunching.

---

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

### Learning: Resonant Field Plasticity (RFP)

- **Implemented** for toy next-token (`wfr_rfp.py`, Exp 06; spec [`docs/11-rfp-v0-spec.md`](docs/11-rfp-v0-spec.md)); online/offline-style stepping on frequencies, `phase_bias`, decay.
- Loss balances task CE, \((1-\text{RC})\), and energy; **strong gains and full theory** remain Phase 2+.
- Plasticity adjusts frequencies and phases, not classical weight matrices.

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

### Basic Pattern Formation (Experiment 04 — Phase 0, Test 2)

**Question:** With **no training**, do different synthetic **position orders** (same length) produce **different** aggregated “standing wave” fingerprints — as theory predicts for Phase 0 ([`docs/03-theory.md`](docs/03-theory.md), section 5)?

**Protocol:** `WFRNetwork` (Freq-Balanced), linear warmup on `0..L−1` (50 passes), then **homeostatic regulation turned off**; **L = 4096**. Six input patterns: linear, reverse, mod-64, stride-7 mod L, fixed shuffle, two identical half-blocks. **Signature** (10-D, L2-normalized): mean standing wave in **four quarters** along the sequence, `std(sw)`, mean spike rate per layer, RC.

| Stage | What it checks | PASS threshold |
|-------|----------------|----------------|
| A | Classwise distinguishability | max pairwise cosine **< 0.99999** |
| B | Same pattern → same signature | cosine of two `linear` runs **> 1 − 10⁻⁵** |
| C (optional) | Robustness to phase noise after WPE (σ = 0.02) | cosine(clean, noisy) **> 0.85** |

**STRONG** (stricter separation: max pairwise cosine **< 0.995**) was **not** achieved on the reference run — some pairs (notably stride vs shuffle) stay almost collinear. The **`mod64`** pattern is the main outlier and sets the **minimum** off-diagonal cosine.

**Reference run (2026-03-31):**

| Metric | Value |
|--------|-------|
| Max off-diagonal cosine | 0.999934 |
| Min off-diagonal cosine | 0.897 |
| Repeatability (`linear`) | 1.0 |
| Clean vs noisy phases (optional) | ~0.99998 |

**Takeaway:** PASS (A+B) and optional (C) support the Phase 0 claim that class signatures **do not collapse** to a single vector; they do **not** imply strong pairwise separation for every class pair without training. Details, heatmap, and JSON: [`experiments/04-basic-pattern-formation/README.md`](experiments/04-basic-pattern-formation/README.md).

### What is not confirmed

- **Time complexity** remains ~O(n), not sub-linear as theorized
- **Learning (RFP)** — forward + toy training tested (`wfr_rfp.py`, Exp 05–07); **large-scale** learning and benchmarks are **Phase 2+**
- **Comparison with existing architectures** — not yet conducted

Full results: [`docs/09-memory-complexity-test-plan.md`](docs/09-memory-complexity-test-plan.md)

## Project Structure

```
wfr/                Каноническое ядро (`core.py`, `losses.py`) — `pip install -e .` или корень репо в PYTHONPATH
docs/               Theory and documentation
experiments/        Test code and results
  00-smoke-test/              Smoke Test + phase0_best_config (ядро — в `wfr/`)
  01-memory-complexity-test/  Memory & Complexity Test (up to 100M tokens)
  02-layer-scaling-test/      Layer Scaling Test (up to 32 layers)
  03-long-context-stability/  Long Context Stability Test
  04-basic-pattern-formation/  Basic Pattern Formation (theory, section 5)
  05-rfp-training-sanity/      Phase 1: training sanity (precheck, full L; short GPU run 2026-03-31 PASS)
  06-rfp-v0/                   RFP v0–v0.3, A/B and grids
  06-rfp-protocol-tests/       Fresh-train / fixed-val protocol + tier checks
  07-wfra2-precursor/          Composite loss + warm-up (WFRA-2 precursor)
data/hf/                     Small real corpus (WikiText-2) — see README there
tools/              Utilities (interactive visualizer)
```

From the repo root, run **`pip install -e .`** for an editable install of the `wfr` package plus `wfr_lm` / `wfr_rfp`, so imports always track your working tree.

## Documentation

1. [`docs/00-overview.md`](docs/00-overview.md) — Architecture overview
2. [`docs/02-architecture.md`](docs/02-architecture.md) — Components and data flow
3. [`docs/08-phase-0-plan.md`](docs/08-phase-0-plan.md) — Phase 0 master plan (viability, exit criteria)
4. [`docs/03-theory.md`](docs/03-theory.md) — Mathematical foundations + v2.0 stability mechanisms + **§10 (tokens in WFRLM / wave-fractal chain)**
5. [`docs/09-memory-complexity-test-plan.md`](docs/09-memory-complexity-test-plan.md) — Test results with v1.0 vs v2.0 comparison
6. [`experiments/04-basic-pattern-formation/README.md`](experiments/04-basic-pattern-formation/README.md) — Basic Pattern Formation (Phase 0, Test 2): protocol, metrics, verdict
7. [`docs/10-phase-1-plan.md`](docs/10-phase-1-plan.md) — Phase 1 (training, RFP): criteria, caveats, experiment registry
8. [`experiments/05-rfp-training-sanity/README.md`](experiments/05-rfp-training-sanity/README.md) — Phase 1 sanity (gradients, full \(L\), val); short run recorded in [`docs/07-experiment-plan.md`](docs/07-experiment-plan.md) Test 3

## License

**AGPL-3.0** — see [LICENSE](LICENSE).

All commits are GPG-signed for verified authorship.

---

Development actively uses neural network models (LLMs) as tools for research, code generation, analysis, and documentation.

**Principle:** Experimental honesty matters more than confirming a hypothesis.
