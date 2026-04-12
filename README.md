# MNESIS: Working Memory in Recurrent Spiking Neural Networks with Heterogeneous Synaptic Delays

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the notebook and paper for MNESIS, a recurrent spiking neural network (SNN) with heterogeneous delays for working memory tasks.

## Paper

**Laurent U. Perrinet** (2026). *Working Memory in a Recurrent Spiking Neural Networks With Heterogeneous Synaptic Delays*. AIROV 2026.

- Paper: [`Perrinet26mnesis.tex`](Perrinet26mnesis.tex)
- Preprint: [`AA_82_Perrinet.pdf`](AA_82_Perrinet.pdf)

## Notebook

**[`MNESIS_polychronous-chains.ipynb`](MNESIS_polychronous-chains.ipynb)** - Complete implementation with parameter sweeps and visualizations.

---

## Principles and Introduction

### Working Memory Challenge

Working memory—the ability to store and recall precise temporal patterns of neural activity—remains a challenge for spiking neural networks. Traditional approaches like gated recurrent units (GRUs) or Transformers rely on rate-based representations, but SNNs must handle long-range temporal dependencies using discrete spikes with millisecond precision.

### Polychronous Groups

- **Synfire chains**: Feedforward pools where synchronous volleys propagate reliably
- **Polychronous neuronal groups (PNGs)**: Time-locked, non-synchronous firing patterns introduced by Izhikevich (2006) with heterogeneous axonal delays
- **Spiking Motifs (SMs)**: Precisely timed spike patterns that generalize PNGs to a trainable framework

### Key Hypothesis

A recurrent SNN with heterogeneous synaptic delays can represent arbitrary spike patterns as **sequential chains of overlapping Spiking Motifs**, where each context window of length *D* uniquely predicts spikes at the next time step, enabling working memory through temporal prediction.

---

## Methods

### Network Architecture

**Recurrent Heterogeneous-Delay SNN (HD-SNN)**:
- **Neurons**: *N* = 512 Leaky Integrate-and-Fire (LIF) neurons
- **Delays**: *D* = 41 discrete delay channels per synapse (1-41 ms)
- **Duration**: *T* = 1000 time steps (1 second total)
- **Parameters**: Weight tensor **W** ∈ ℝ^(N×N×D) ≈ 10^7 parameters

### Membrane Dynamics

```
u_j(t) = β·u_j(t-1)·(1 - s_j(t-1)) + Σ_i Σ_d W_{j,i,d}·s_i(t-d)
```

where:
- β ∈ (0,1) is the membrane decay (initialized to 0.7, learned)
- s_j(t) ∈ {0,1} is the spike of neuron j at time t
- W_{j,i,d} is the synaptic weight from neuron i to j at delay d
- Spike emitted when u_j(t) > ϑ = 1, then membrane reset

### Training Objective

**F1-Score Loss**: L = 1 - F₁

where F₁ is the harmonic mean of:
- **Precision**: TP / (TP + FP) — penalizes over-activity
- **Recall**: TP / (TP + FN) — penalizes silence

This symmetrically prevents two failure modes: silent networks and over-active networks.

### Weight Initialization

Analytical pseudo-inverse initialization:

```
w_{ij}^(d) ∝ Σ_μ Σ_t s_j^*(μ)(t) · s_i^*(μ)(t-d)
```

Equivalent to Hebbian-like learning averaged over all stored patterns.

### Training Procedure

1. Generate *M* = 8 random target patterns (Bernoulli, p = 10⁻³)
2. Clamp initial window: t ∈ [0, D) to target values
3. Forward pass through recurrent network
4. Compute F1 loss between prediction and target
5. Backpropagate through time using surrogate gradients (fast-sigmoid, α = 20)
6. Optimize with Adam (lr = 10⁻⁶, momentum = 0.1, weight decay = 10⁻³)
7. Cosine learning rate schedule for 4096 steps

### Implementation

- **Framework**: snnTorch + PyTorch
- **Hardware**: Apple M3 Ultra (MPS) and CUDA (Jean Zay GENCI)
- **Training time**: ~10 minutes per run
- **Code**: `MNESIS_polychronous-chains.ipynb`

---

## Results

### Performance

| Metric | Value |
|--------|-------|
| Mean F1 Score | 0.966 |
| Patterns stored | M = 8 |
| Network size | N = 512 neurons |
| Pattern duration | T = 1000 steps (1 s) |
| Parameters | N² × D ≈ 10^7 |

### Sequential Learning

Training exhibits **temporal progression**:
- Correct recall emerges first immediately after clamped window
- Learning extends progressively toward later time steps
- Errors at early steps propagate to later predictions

![Target](figures/target.pdf)

### Parameter Effects

![Parameter scans](figures/MNESIS_num_delay.pdf)

**Delay depth D** (primary capacity lever):
- Loss decreases monotonically with D
- D = 3: L ≈ 0.85
- D = 127: L → 0
- Larger D provides more orthogonal context windows

**Pattern duration T** (primary difficulty lever):
- Loss grows with T
- T = 64: L ≈ 0.004
- T = 2048: L ≈ 0.08
- Longer sequences compound prediction errors

**Firing rate p_A** (orthogonality regime):
- Optimal: p_A ∈ [10⁻⁴, 10⁻³]
- Degraded: p_A ≥ 2×10⁻³
- Higher rates increase inter-pattern interference

### Key Findings

1. **Heterogeneous delays are computational asset**: The temporal structure introduced by D delay channels enables compression—storing N×T bits with N²×D parameters

2. **Delay depth > network size for capacity**: Increasing D improves both parameter capacity and context orthogonality, while adding no neurons

3. **Sequential prediction from context**: Each window [t-D, t-1] predicts activity at time t, implementing memory as chain of temporal predictions

---

## Figures

| Figure | Description |
|--------|-------------|
| [`figures/izhikevich_rec.pdf`](figures/izhikevich_rec.pdf) | Working memory as sequential spike prediction |
| [`figures/unrolled.pdf`](figures/unrolled.pdf) | Unrolled recurrent HD-SNN architecture |
| [`figures/pattern.pdf`](figures/pattern.pdf) | Example target spike pattern |
| [`figures/target.pdf`](figures/target.pdf) | Training dynamics raster plot |
| [`figures/MNESIS_num_delay.pdf`](figures/MNESIS_num_delay.pdf) | Effect of delay depth |
| [`figures/MNESIS_N_time.pdf`](figures/MNESIS_N_time.pdf) | Effect of pattern duration |
| [`figures/MNESIS_p_A.pdf`](figures/MNESIS_p_A.pdf) | Effect of firing rate |

---

## Getting Started

### Requirements

```bash
pip install torch snntorch numpy matplotlib pandas
```

### Run

```bash
jupyter notebook MNESIS_polychronous-chains.ipynb
```

### Cached Data

Results cached in `cached_data/`:
- `*_init.pth`: Pseudo-inverse initialized weights
- `*.pth`: Trained model weights
- `*_scan_*.json`: Parameter sweep results

---

## Citation

```bibtex
@inproceedings{Perrinet2026MNESIS,
  author = {Perrinet, Laurent U.},
  title = {Working Memory in a Recurrent Spiking Neural Networks 
           With Heterogeneous Synaptic Delays},
  booktitle = {AIROV 2026},
  year = {2026},
  url = {https://github.com/laurentperrinet/MNESIS}
}
```

## License

MIT License - see [LICENSE](LICENSE)