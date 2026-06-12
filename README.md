# MNESIS — Working Memory in a Recurrent Spiking Neural Network with Heterogeneous Synaptic Delays

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-AIROV%202026-orange.svg)](tex/Perrinet26mnesis.pdf)

> **MNESIS** — *Memory Network Every Spike Is Sacred*

This repository contains the full implementation, experiments, and paper source for **MNESIS**, a recurrent spiking neural network (SNN) with heterogeneous synaptic delays that stores and recalls arbitrary spike patterns as sequential chains of overlapping Spiking Motifs.

---

## Overview

Working memory in biological neural circuits relies on precise spike timing rather than sustained firing rates. MNESIS models this by equipping every synapse with $D = 41$ learnable delays, parameterised as a single weight tensor $\mathbf{W} \in \mathbb{R}^{N \times N \times D}$. Each stored pattern is encoded as a chain of overlapping **Spiking Motifs**: contiguous context windows of length $D$ that uniquely predict the next time step of activity. A closed-form Hebbian initialisation derived by deconvolving the LIF membrane response achieves perfect recall ($F_1 = 1.0$) before any gradient step, while surrogate-gradient BPTT provides robustness to noise.

**Key results:**
- Perfect recall of $M = 16$ patterns of $T = 1000\,\mathrm{ms}$ with $N = 1024$ neurons
- $F_1 = 1.0$ from analytical initialisation alone — no gradient step required
- Tolerates 25% bit-flip noise in the trigger window (attractor dynamics)
- Memory capacity scales as $N^2 D$, linear in delay depth

**Paper:** Laurent U. Perrinet (2026). *Working Memory in a Recurrent Spiking Neural Network with Heterogeneous Synaptic Delays*. AIROV 2026.
[`tex/Perrinet26mnesis.pdf`](tex/Perrinet26mnesis.pdf) · [`https://laurentperrinet.github.io/publication/perrinet-26-icann/`](https://laurentperrinet.github.io/publication/perrinet-26-icann/)

---

## Repository structure

```
MNESIS/
├── src/                          # All Jupyter notebooks (numbered pipeline)
│   ├── 10_MNESIS_polychronous-chains.ipynb   # Model definition and core mechanism
│   ├── 11_MNESIS_learn-synthetic.ipynb       # Training on synthetic patterns
│   ├── 13_MNESIS_testing-inference.ipynb     # Sequential retrieval of M patterns
│   ├── 14_MNESIS_testing-noise.ipynb         # Robustness to bit-flip noise
│   ├── 15_MNESIS_testing-trigger-duration.ipynb  # Effect of trigger-window length
│   ├── 16_MNESIS_testing-trigger-fraction.ipynb  # Effect of partial neuron coverage
│   ├── 20_MNESIS_scanning-parameters.ipynb   # Parameter scans (D, T, p_A, ...)
│   └── 25_MNESIS_optuna.ipynb                # Hyperparameter optimisation (Optuna)
├── figures/                      # Generated figures (PDF/PNG)
├── tex/                          # Paper source
│   ├── Perrinet26mnesis.tex      # Main LaTeX source
│   ├── Perrinet26mnesis.pdf      # Compiled paper
│   ├── mnesis.bib                # Bibliography
│   ├── llncs.cls                 # Springer LNCS class
│   └── splncs04.bst              # BibTeX style
├── cached_data/                  # Cached weights and scan results (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `snntorch`, `numpy`, `scipy`, `matplotlib`, `jupyter`.

### Run the notebooks in order

The notebooks are numbered and designed to be run sequentially. Each notebook saves its outputs (model weights, scan results) to `cached_data/` so that downstream notebooks can load them without recomputation.

```bash
cd src
jupyter notebook
```

| # | Notebook | Purpose |
|---|----------|---------|
| 10 | `10_MNESIS_polychronous-chains.ipynb` | Define the `HD_SNN` class, generative model, and the analytical weight initialisation (Hebbian cross-correlation with LIF deconvolution). Corresponds to the Methods section of the paper. |
| 11 | `11_MNESIS_learn-synthetic.ipynb` | Train the network on $M = 16$ synthetic sparse patterns. Demonstrates that analytical init alone reaches $F_1 = 1.0$; gradient training then improves noise robustness. |
| 13 | `13_MNESIS_testing-inference.ipynb` | Concatenate all $M = 16$ patterns in sequence with spontaneous inter-trial intervals; evaluate sliding-window $F_1$ to confirm selective, cross-interference-free retrieval. |
| 14 | `14_MNESIS_testing-noise.ipynb` | Bit-flip noise on the trigger window ($p_\mathrm{flip} \in [0, 1]$). Quantifies attractor-like robustness of retrieval. |
| 15 | `15_MNESIS_testing-trigger-duration.ipynb` | Truncated trigger window (0 to $D-1$ steps). Finds the minimum cue length for reliable recall. |
| 16 | `16_MNESIS_testing-trigger-fraction.ipynb` | Partial neuron coverage (0 to $N$ neurons silenced in trigger). Quantifies population-size tolerance. |
| 20 | `20_MNESIS_scanning-parameters.ipynb` | Systematic one-at-a-time scans over $D$, $T$, $p_A$, $N$, $E_S$, $p_\mathrm{SM}$, etc. with $N_\mathrm{cv} = 10$ seeds. Produces the parameter-scan figures. |
| 25 | `25_MNESIS_optuna.ipynb` | Automated hyperparameter search with [Optuna](https://optuna.org/) for learning rate, dropout, surrogate sharpness, and momentum. |

### Cached data

Results are saved to `cached_data/` (excluded from git via `.gitignore`):

| File pattern | Content |
|---|---|
| `*_init.pth` | Analytically initialised weights (pseudo-inverse or Hebbian) |
| `*.pth` | Trained model weights after gradient steps |
| `*_scan_*.json` | Parameter sweep results (loss, precision, recall per condition) |

Delete a `.pth` or `.json` file to force recomputation; set `RECOMPUTE = True` at the top of any notebook to invalidate the full cache for that notebook.

---

## Model

### Membrane dynamics

Each neuron $j$ evolves as:

$$u_j(t) = \beta \cdot u_j(t-1) \cdot (1 - s_j(t-1)) + \sum_{i=1}^{N} \sum_{d=1}^{D} W_{j,i,d} \cdot s_i(t-d)$$

with $\beta = 0.7$ ($\tau \approx 2.8\,\mathrm{ms}$), threshold $\vartheta = 1$, and zero-reset after each spike.

### Analytical initialisation

The LIF membrane is a causal IIR lowpass $H(z) = 1/(1 - \beta z^{-1})$. The optimal input current that places the membrane at a sub-threshold target $\vartheta_0 = 0.8$ at each target spike time is obtained by deconvolution:

$$I^*_j(t) = \vartheta_0 \bigl(s^*_j(t) - \beta \cdot s^*_j(t-1)\bigr)$$

The closed-form initialisation follows from the Gram-matrix approximation $\mathbf{C}\mathbf{C}^\top \approx N D p_A \mathbf{I}$:

$$w_{i,j,d} = \frac{\vartheta_0}{N \cdot D \cdot p_A \cdot M} \sum_{\mu,\,t} s_i^{*(\mu)}(t-d) \cdot \bigl(s_j^{*(\mu)}(t) - \beta \cdot s_j^{*(\mu)}(t-1)\bigr)$$

The safety margin $\delta = \vartheta - \vartheta_0 = 0.2$ maximises surrogate gradient sensitivity at initialisation ($\sigma'_{15}(-0.2) \approx 0.94$) while preventing spurious spikes from partial contexts.

### Training

- **Loss**: $\mathcal{L} = 1 - F_1$ (harmonic mean of precision and recall, evaluated after the trigger window)
- **Optimiser**: AdamW, cosine schedule with linear warm-up
- **Surrogate**: fast sigmoid, sharpness $\alpha = 15$
- **Regularisation**: dropout $p = 0.37$, weight decay $\lambda = 0$
- **Hardware**: Apple M3 Ultra (MPS) or NVIDIA GPU (CUDA / Jean Zay GENCI)

---

## Results summary

| Experiment | Key result |
|---|---|
| Training (NB 11) | $F_1 = 1.0$ with Hebbian init alone; gradient training adds noise robustness |
| Sequential retrieval (NB 13) | All 16 patterns retrieved without cross-interference |
| Noise robustness (NB 14) | $F_1 = 0.967$ at $p_\mathrm{flip} = 0.25$; chance only near $p_\mathrm{flip} = 0.5$ |
| Trigger duration (NB 15) | $F_1 = 0.862$ at 75% of $D$; reliable above $D/2$ |
| Neuron coverage (NB 16) | $F_1 = 1.0$ with 87.5% of neurons active in trigger |
| Delay scan (NB 20) | $\mathcal{L} \approx 0.85$ at $D=3$; $\mathcal{L} \to 0$ at $D=127$ |
| Duration scan (NB 20) | $\mathcal{L} \approx 0.004$ at $T=64$; $\mathcal{L} \approx 0.08$ at $T=2048$ |
| Rate scan (NB 20) | Optimal at $p_A \in [10^{-4}, 10^{-3}]$; degrades for $p_A \geq 2\times10^{-3}$ |

---

## Citation

```bibtex
@inproceedings{Perrinet2026MNESIS,
  author    = {Perrinet, Laurent U.},
  title     = {Working Memory in a Recurrent Spiking Neural Network
               with Heterogeneous Synaptic Delays},
  booktitle = {AIROV 2026},
  year      = {2026},
  url       = {https://laurentperrinet.github.io/publication/perrinet-26-icann/}
}
```

---

## License

GPL-3.0 — see [LICENSE](LICENSE).

---

*Institut de Neurosciences de la Timone (UMR 7289), Aix Marseille Université / CNRS, Marseille, France.*
*Supported by GENCI-IDRIS (Grant 2025–AD010314955R2).*
