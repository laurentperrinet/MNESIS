# MNESIS: Memory Network Every Spike Is Sacred

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue.svg)](https://doi.org/xxxxx)

A collection of spiking neural network (SNN) simulations exploring distributed memory networks with axonal delays. MNESIS implements various approaches to modeling recurrent neural networks where each spike is treated as a discrete event with transmission delays.

## Overview

This repository contains implementations of spiking neural networks that investigate fundamental questions in computational neuroscience:

- **Distributed memory encoding** - How memories form through spike patterns across neural populations
- **Temporal coding** - Information encoding and processing through precise spike timing
- **Network dynamics** - Self-sustained activity, synchronization, and stability in recurrent networks
- **Biological realism** - Incorporation of axonal conduction delays (2-40ms) and leaky integrate-and-fire neuron dynamics

The "Every Spike Is Sacred" philosophy emphasizes treating each neural spike as a discrete, meaningful event that contributes to memory formation and information processing in distributed neural systems.

## Implementation Frameworks

MNESIS implements spiking neural networks using multiple simulation approaches:

### Event-Based Spiking Networks
- Multi-threaded SNN where each neuron runs as a separate thread
- Uses priority queues for precise timing of delayed spikes
- Implements Poisson spike generation and delayed spike transmission
- Scalable from small networks (32 neurons) to large ones (8192+ neurons)

### Standard Neuroscience Frameworks
- **NumPy-based SNN** - Custom LIF neuron implementation using NumPy arrays
- **Brian2** - Implementation using the Brian2 SNN simulation library
- **PyNN** - Standardized neural simulations using PyNN interface

### Modern Deep Learning Integration
- **snnTorch** - Integration with snnTorch for hybrid SNN-deep learning approaches
- **HD encoding** - High-dimensional computing integration with spiking networks

### Networking Approaches
- ZeroMQ-based implementations for inter-process communication
- Threading vs multiprocessing performance comparisons
- Scalability tests across network sizes

## Notebook Examples

### Core Implementations

| Notebook | Description | Key Features |
|----------|-------------|--------------|
| `2026-01-21 event-based SNN.ipynb` | Multi-threaded event-based SNN | Priority queues, delayed transmission, wall-clock timing |
| `2026-01-21 numpy-based SNN.ipynb` | Custom LIF neuron implementation | Complete control over neuron dynamics, spike propagation |
| `2026-01-21 brian-based SNN.ipynb` | Brian2 library implementation | Standardized SNN simulation, rapid prototyping |
| `2025-09-25 polychronous chain.ipynb` | Polychronous neural groups | Temporal pattern formation, synfire chains |

### Advanced Architectures

| Notebook | Description | Applications |
|----------|-------------|--------------|
| `2026-02-02_snnTorch.ipynb` | snnTorch integration | Hybrid SNN-deep learning, gradient-based training |
| `2026-02-05_HD-snnTorch.ipynb` | High-dimensional computing | Binary vector embeddings, pattern recognition |

## Key Features

###Neural Modeling
- Leaky Integrate-and-Fire (LIF) neurons with biophysically realistic parameters
- Variable axonal delays (2-40ms range) for conduction timing
- Poisson spike generation with configurable base firing rates (~1Hz)
- Sparse connectivity patterns with configurable connection density (~20%)

### Technical Innovations
- Thread-safe priority queues for precise spike timing
- ZeroMQ integration for distributed communication between network components
- Memory-efficient spike tracking and counting mechanisms
- Real-time simulation with wall-clock timing synchronization

### Scalability & Performance
- Seamless scaling from small networks (32 neurons) to large-scale simulations (8192+ neurons)
- Multi-core processing support via threading and multiprocessing
- Memory-optimized data structures for large-scale network simulations
- Performance benchmarking across different implementation approaches

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -U -r requirements.txt
```

Core dependencies include:
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `torch` + `snnTorch` - Deep learning integration
- `jupyter` + `ipywidgets` - Interactive notebooks
- Additional packages: `seaborn`, `cmocean`, `pandas`

### Running Simulations

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MNESIS
   ```

2. Install dependencies:
   ```bash
   pip install -U -r requirements.txt
   ```

3. Launch Jupyter notebook:
   ```bash
   jupyter notebook
   ```

4. Open any notebook file and execute cells sequentially
5. Modify parameters to explore different network configurations:

   - Network size: 32-8192+ neurons
   - Connectivity density: 5-50%
   - Axonal delays: 2-40ms
   - Firing rates: 0.5-5Hz

## Project Structure

```
MNESIS/
├── *.ipynb              # Jupyter notebooks with complete SNN implementations
│   ├── 2025-09-25 polychronous chain.ipynb
│   ├── 2026-01-21 event-based SNN.ipynb
│   ├── 2026-01-21 numpy-based SNN.ipynb
│   ├── 2026-01-21 brian-based SNN.ipynb
│   ├── 2026-02-02_snnTorch.ipynb
│   └── 2026-02-05_HD-snnTorch.ipynb
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

Each notebook contains executable code with detailed explanations, parameter descriptions, and visualization code for analyzing network activity.

## Research Applications

This research toolkit enables investigation of:

### Distributed Memory Systems
- Pattern storage and recall in recurrent networks
- Memory capacity analysis across network sizes
- Robustness of memory retrieval to noise and neuron loss

### Temporal Processing
- Spike-timing dependent plasticity (STDP) effects
- Temporal pattern recognition and classification
- Network responses to precisely timed inputs

### Network Dynamics
- Emergent synchronization and oscillations
- Balanced excitation/inhibition dynamics
- Criticality and power-law distributions in activity

## Citation

If you use this code in your research, please cite:

```bibtex
@software{MNESIS2026,
  author = {Author Name},
  title = {MNESIS: Memory Network Every Spike Is Sacred},
  year = {2026},
  publisher = {GitHub},
  url = {<repository-url>}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work contributes to the computational neuroscience community's efforts to understand how temporal dynamics and distributed representations enable memory and computation in biological and artificial neural systems.
