# AGENTS.md — Developer & AI-Assistant Reference

## Project Summary

MNESIS is a recurrent spiking neural network (SNN) with heterogeneous synaptic delays. The core research code lives in two Python modules (`mnesis_boilerplate.py`, `mnesis_chains.py`); experiment notebooks import from these modules and are numbered to run sequentially.

An AI assistant was used to improve the readability of the code, not to create it - theis file describes the rules used to guide the formatting and testing of the code.

---

## Architecture Overview

### Module Dependency Graph

```
notebooks (*.ipynb)
  └── mnesis_chains.py        (HD_SNN, SpikingPattern, StochasticSpikingPattern, load())
        └── mnesis_boilerplate.py   (imports, Params dataclass, utilities, SpikeF1scoreLoss)
              └── snntorch / torch / numpy / matplotlib / ...
```

### Key Source Files (`src/`)

| File | Role |
|---|---|
| `mnesis_boilerplate.py` | Entry-point imports (torch, snntorch, etc.), device auto-detection (MPS → CUDA → CPU), `Params` dataclass with all hyperparameters, utility functions (`flip_bits`, `printfig`, `get_scores`, `SpikeF1scoreLoss`, cosine LR schedule). This used to live in notebooks 01 and 05. |
| `mnesis_chains.py` | Pattern generators (`SpikingPattern`, `StochasticSpikingPattern`) and the network class `HD_SNN` (constructor, `forward_pass`, `get_W_init`, `update_weight`, `learn_model`). Also exports a `load()` helper for loading trained checkpoints. This used to live in notebook 08 and 10. |

### Notebooks (`src/*.ipynb`)

Notebooks are numbered sequentially from simple to more complex and each depends on the previous ones (via cached artifacts). Each notebook is an experiment for testing the network. They **import** from `.py` modules — they do **not** contain standalone copies of `Params`, `HD_SNN`, etc.

| Prefix range | Purpose |
|---|---|
| 10–16 | Core experiments: generative model, training, inference, noise/trigger robustness |
| 20–25 | Sweeps & hyperparameter optimisation (grid scan + Optuna) |
| 30–40 | Extended pattern types: periodic, travelling waves, Lorenz attractor, SHD dataset |
| 99 | End-to-end orchestrator — runs the full pipeline with progress timestamps |

> **NOTE**: Notebooks 01 (`boilerplate`), 05 (`parameters`), and 08 (`generative-model`) no longer exist as notebooks. Their code has been refactored into `mnesis_boilerplate.py` and `mnesis_chains.py`. Any references to these notebook numbers in old documents are stale.

---

## Running the Code

### Prerequisites

```bash
pip install -r src/requirements.txt
```

Required: Python 3.10+, PyTorch, snntorch, jupyter, scipy, matplotlib, seaborn, optuna, MotionClouds, scikit-learn, h5py.

### Device Selection

`mnesis_boilerplate.py` auto-detects hardware. Priority order: MPS (Apple Silicon) → CUDA (NVIDIA GPU) → CPU. On the Jean Zay cluster (`USER == "uvb28bo"`), figure saving is disabled automatically.

### Debug Mode

Set `DEBUG > 1` in `mnesis_boilerplate.py` to reduce problem size:
- `N_neuron`, `N_pattern`, `N_time`, and `num_epochs` are divided by `DEBUG`
- Set `DEBUG = 1` for production runs (default)

### Cache Invalidation

Artifacts are cached in `cached_data/`. To recompute:
- Delete individual `.pth`, `.json`, `.npz`, or `.sqlite3` files, or
- Set `RECOMPUTE = True` at the top of the notebook to invalidate the full cache


FILE-BASED LOCKING MECHANISM:
 - Creates a `.lock` sentinel file to prevent concurrent execution of the same scan
 - Workflow: 
   1. If RECOMPUTE=True, delete both data and lock files to force restart
   2. Load existing results if available
   3. If no lock file exists, create one and start processing
   4. For each parameter value, check if already computed; if not, run scan and append to JSON
   5. Delete lock file when done
 - Purpose: Prevents multiple instances from running the same scan parameter simultaneously,
   allowing safe resumption and incremental computation without data corruption


---

## Params Dataclass (Live Reference)

Defined in `mnesis_boilerplate.py`. Key fields and current defaults:

| Field | Default | Notes |
|---|---|---|
| `N_neuron` | `1024 // DEBUG` | Presynaptic inputs |
| `num_delay` | `41` | Must be odd (convolution symmetry) |
| `N_pattern` | `16 // DEBUG` | Number of spiking motifs |
| `N_time` | `1000 // DEBUG` | Timebins per WM pattern |
| `N_pretime` | `50` | Spontaneous activity before/after stimulus |
| `p_A` | `0.00016` | Prior firing probability |
| `lif_beta` | `0.8` | Membrane decay (was 0.7 in paper text) |
| `lif_threshold` | `0.80` | Spike threshold (was 1.0 in paper text) |
| `alpha_surrogate` | `12.0` | Surrogate sharpness (was 15 in paper) |
| `dropout` | `0.10` | Dropout rate (was 0.37 in paper) |
| `optimizer` | `"sgd"` | Options: `sgd`, `adam`, `adamw`, `rmsprop`, `adadelta` |
| `loss_name` | `"SpikeF1scoreLoss"` | Alternative: `"MSELoss"` |
| `reset_mechanism` | `"subtract"` | Alternative: `"zero"` |

### ⚠️ Parameter Drift Warning

Some default values in `mnesis_boilerplate.py` differ from the paper text and README (e.g. $\beta = 0.8$ vs 0.7, $\vartheta = 0.8$ vs 1.0). The code defaults represent the **latest camera-ready run** (`datetag = '2026-07-11'`). If reproducing paper figures, check which value was actual used.

---

## Code Conventions

- Imports are at the top of `mnesis_boilerplate.py`; downstream modules re-export via `from mnesis_boilerplate import ...`
- No virtual environment is committed; `.venv/` is git-ignored
- Figure output goes to `../figures/` (relative from `src/`); disabled on Jean Zay
- `datetag` controls cache filename prefix — change it to start a new experiment batch
- `seed = 2018` is fixed for reproducibility
