# MNESIS : Memory Network Every Spike Is Sacred

MNESIS is a collection of spiking neural network (SNN) simulations exploring distributed memory networks with axon delays. The project implements various approaches to modeling recurrent neural networks where each spike is treated as a discrete event with transmission delays.

## Project Overview

This project implements multiple SNN simulations that explore:
- **Event-based spiking networks** with realistic axon delays (2-40ms)
- **Different simulation frameworks** including threading, ZeroMQ, NumPy, Brian2, and PyNN
- **Self-sustained activity** in recurrent networks
- **Distributed memory** concepts where information is encoded in spike patterns

## Notebook Examples

### Event-Based Spiking Networks
- `2026-01-21 event-based SNN.ipynb` - Multi-threaded SNN where each neuron runs as a separate thread
- Implements Poisson spike generation and delayed spike transmission
- Uses priority queues for precise timing of delayed spikes

### Networking Approaches
- ZeroMQ-based implementations for inter-process communication
- Threading vs multiprocessing comparisons
- Scalability tests from 32 to 8192 neurons

### Neuroscience Frameworks
- `2026-01-21 numpy-based SNN.ipynb` - Custom LIF neuron implementation using NumPy
- `2026-01-21 brian-based SNN.ipynb` - Implementation using Brian2 SNN library
- PyNN-based approaches for standardized neural simulations

### Specialized Networks
- `2025-09-25 polychronous chain.ipynb` - Exploration of polychronous neural groups

## Key Features

### Realistic Neural Modeling
- Leaky Integrate-and-Fire (LIF) neurons
- Variable axonal delays (2-40ms range)
- Poisson spike generation with ~1Hz base rate
- Sparse connectivity with ~20% density

### Technical Innovations
- Thread-safe priority queues for spike timing
- ZeroMQ integration for distributed communication
- Memory-efficient spike tracking and counting
- Real-time simulation with wall-clock timing

### Scalability
- Scalable from small networks (32 neurons) to large ones (8192+ neurons)
- Multi-core processing support via threading/multiprocessing
- Memory-optimized data structures

## Applications

This research explores fundamental questions in:
- **Distributed memory encoding** - How memories form through spike patterns
- **Temporal coding** - Information encoding in spike timing
- **Network dynamics** - Self-sustained activity and synchronization
- **Computational neuroscience** - Models of biological neural computation

## Getting Started

### Prerequisites
```bash
pip install numpy matplotlib
pip install brian2 pyNN
pip install zmq
```

### Running Simulations
1. Open any notebook file in Jupyter
2. Follow the step-by-step implementation
3. Modify parameters for different network sizes and behaviors

## Project Structure

- `*.ipynb` - Jupyter notebooks with complete implementations
- Each notebook contains executable code with detailed explanations
- Educational examples ranging from simple to advanced implementations

## Research Goals

The "Memory Network Every Spike Is Sacred" philosophy emphasizes treating each neural spike as a discrete, meaningful event contributing to memory formation and information processing in distributed neural systems.

![ESIS](https://static1.srcdn.com/wordpress/wp-content/uploads/2020/03/Every-Sperm-is-Sacred.jpg)