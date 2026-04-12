# MNESIS Polychronous Chains

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Accurate detection and generation of polychronous spike motifs using spiking neural networks with snnTorch.

## Introduction

Polychronous chains are temporally precise spike patterns that emerge in recurrent neural networks with axonal delays. This notebook implements a spiking neural network (SNN) capable of detecting and generating such patterns using gradient-based learning with snnTorch.

## Principles

### Polychronous Groups
- Groups of neurons that fire in a time-locked but not synchronous manner
- Activity propagates through the network via axonal delays
- Different neurons fire at different times, creating temporal spike motifs

### Spike Motifs as Memory
- Each spiking motif (SM) represents a distributed memory pattern
- The network learns to reproduce target spike patterns from initial conditions
- Temporal precision enables dense information storage

### Key Hypothesis
A spiking neural network with appropriately initialized recurrent weights can learn to propagate polychronous chains from an initial seed pattern, enabling controlled pattern generation and memory retrieval.

## Methods

### Network Architecture
- **Neuron Model**: Leaky Integrate-and-Fire (LIF) with surrogate gradients
- **Input Layer**: Flattened spike windows of size `num_delay × N_neuron`
- **Hidden Layer**: Dense recurrent connections with dropout regularization
- **Output**: Spike trains for each neuron over time

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_neuron` | 256 | Number of presynaptic neurons |
| `num_delay` | 41 | Temporal window size (timesteps) |
| `N_SM` | 4 | Number of spiking motifs |
| `N_time` | 2000 | Total simulation duration (timesteps) |
| `p_A` | 0.001 | Spontaneous firing probability |
| `lif_beta` | 0.7 | Membrane potential decay rate |

### Weight Initialization
Weights are initialized using a pseudo-inverse approach:
1. Collect context-target spike pairs from target patterns
2. Compute Moore-Penrose pseudo-inverse of context matrix
3. Set weights to minimize reconstruction error
4. Apply gain factor for spike generation

### Loss Function
**Spike F1-Score Loss**: Balances precision and recall for spike prediction
- Precision: TP / (TP + FP) - minimizes false positives
- Recall: TP / (TP + FN) - minimizes false negatives
- F1 Score: Harmonic mean of precision and recall

### Training Procedure
1. Generate random target spike patterns
2. Initialize input spikes with spontaneous activity + target seed
3. Forward pass through LIF network with recurrent connections
4. Compute F1-score loss between predicted and target spikes
5. Backpropagate through time using surrogate gradients
6. Update weights with SGD or Adam optimizer
7. Use cosine learning rate schedule with warmup

## Results

### Performance
- Training achieves **near-perfect F1 scores** (precision=1.0, recall=1.0) on target pattern reproduction
- The network successfully propagates spike patterns through time using learned recurrent weights
- Initial pseudo-inverse weights provide strong baseline performance

### Spike Pattern Generation
- Input: Initial spike seed (first `num_delay` timesteps of target pattern) + spontaneous activity
- Output: Full target spike pattern reproduced by the recurrent network
- The network acts as a **pattern completion system** - from seed to full motif

### Parameter Scans
The notebook systematically explores:
- **Number of patterns (N_SM)**: How many distinct motifs can be stored
- **Pattern duration (N_time)**: Temporal extent of spike patterns
- **Maximum delay (num_delay)**: Range of axonal delays
- **Network size (N_neuron)**: Scaling with number of neurons
- **Spontaneous firing rate (p_A)**: Noise robustness

### Visualizations
- Raster plots showing target vs predicted spike patterns
- Weight distribution histograms
- Training loss and F1-score curves

## Getting Started

### Requirements
```bash
pip install torch snntorch numpy matplotlib
```

### Run the Notebook
```bash
jupyter notebook MNESIS_polychronous-chains.ipynb
```

### Cached Data
Training results are cached in `cached_data/` directory:
- `*_init.pth`: Initial weights from pseudo-inverse
- `*.pth`: Trained model weights
- `*.json`: Parameter scan results

## Citation

```bibtex
@software{MNESIS_polychronous2026,
  author = {Laurent Perrinet},
  title = {MNESIS Polychronous Chains: Spike Motif Detection with SNNs},
  year = {2026},
  publisher = {GitHub},
  url = {<repository-url>}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.