from pathlib import Path
from dataclasses import dataclass, asdict, field
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
import snntorch as snn
import snntorch.surrogate as surrogate
from snntorch import utils as snn_utils
import snntorch.spikeplot as splt
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
import seaborn as sns
import pandas as pd
import datetime
# --- Configuration & Paths ---
RECOMPUTE = False
DEBUG = 1 # production
if DEBUG > 1:
    print(f'running in debug mode with DEBUG = {DEBUG}')

datetag = '2026-07-11' # new run with new parameters from the camera ready
print(f"datetag = '{datetag}'")

# --- Torch Setup ---
torch.set_float32_matmul_precision("medium")
torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)
torch.autograd.set_detect_anomaly(True)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


@dataclass
class Params:
    """Configuration describing a run of the MNESIS spiking-neural network.

    This used to be defined in the ``05_MNESIS_parameters.ipynb`` notebook. It
    now lives in :mod:`mnesis_boilerplate` so that it can be imported directly
    by every downstream notebook / module (``from mnesis_boilerplate import
    Params, asdict``), matching the behaviour of the original ``%run`` chain.
    """
    datetag: str = datetag  # noqa: TID251 - intentional default from config

    N_neuron: int = 1024 // DEBUG        # number of presynaptic inputs
    num_delay: int = 41                  # number of timesteps in SM, must be a odd number for convolutions
    N_pattern: int = 16 // DEBUG         # number of spiking motifs
    N_time: int = 1000 // DEBUG          # number of timebins for the WM patterns
    N_pretime: int = 50                  # number of timebins for spontaneous activity before and after the stimulus
    p_A: float = 0.00016                 # prior probability of firing for postsynaptic raster plot (spike per timebin)
    p_flip: float = 0.01                 # the default probability of flipping a bit in the stochastic pattern generator
    seed: int = 2018                     # seed
    device = device

    # network
    lif_beta: float = 0.8
    lif_threshold: float = 0.80
    learn_beta: bool = False
    learn_threshold: bool = False
    do_pinv: bool = True
    do_deconv: bool = True

    # learning
    num_epochs: int = 256 // DEBUG
    num_warmup_epochs: int = 16          # 2**4
    base_lr: float = 20.0e-3
    final_lr: float = 400.e-6
    delta1: float = 20.e-3
    delta2: float = 20.e-6
    dropout: float = 0.10
    alpha_surrogate: float = 12.0
    surrogate_name: str = "FastSigmoid"
    loss_name: str = "SpikeF1scoreLoss"  # 'MSELoss' #'L1Loss'
    reset_mechanism: str = "subtract"    # "zero"
    optimizer: str = "sgd"              # 'adamw' #adam

    # figures
    verbose: bool = False                # Displays more verbose output.
    fig_width: float = 15                # width of figure
    fig_height: float = 9                # width of figure
    phi: float = 1.61803                 # beauty is gold
    N_time_show: int = 512               # number of time points to show in plots
    N_neuron_show: int = 200             # number of SM to show in plots
    N_scan: int = 35 // DEBUG + 1        # number of values to scan

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)



data_cache = Path('../cached_data')
data_cache.mkdir(exist_ok=True)

figpath = Path('../figures')
if os.environ.get("USER") == "uvb28bo": 
    figpath = None # Jean Zay


# --- Constants ---
phi = np.sqrt(5)/2 + 1/2
subplotpars = SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)
i_pattern = 0

# --- Utility Functions ---
def pprint(s):
    print(len(s)*'=')
    print(s)
    print(len(s)*'=')

def printfig(fig, name, fig_width, fig_height=None, exts=['pdf', 'png', 'svg'], figpath=figpath, dpi_exp=None, bbox='tight', verbose=True):
    if figpath is not None: 
        figpath.mkdir(exist_ok=True)
        if fig_height is None: fig_height = fig_width/phi
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches((fig_width*cm, fig_height*cm))
        for ext in exts:
            filename = figpath / f'{name}.{ext}'
            if verbose: print(f'Saving as {filename}')
            fig.savefig(filename, dpi=dpi_exp, bbox_inches=bbox, transparent=True)

def flip_bits(a, p_flip, seed=None, verbose=False):
    generator = torch.Generator(device=a.device)
    if seed is None:
        seed = generator.seed()
    else:
        generator.manual_seed(seed)
    mask = torch.bernoulli(torch.ones_like(a) * p_flip, generator=generator)
    if verbose:
        print(f"Flipping {mask.sum().item()} bits out of {a.numel()} (p_flip={p_flip}, seed={seed}), {a.mean().item():.3e} -> {torch.where(mask == 1., 1 - a, a).mean().item():.3e}")
    flipped = torch.bernoulli(torch.ones_like(a) * a.mean(), generator=generator)
    return torch.where(mask == 1., flipped, a)

def stop(): 
    assert False, "Temporary end of the road"

def approx_equals(series, value, rtol=1e-6, atol=1e-12):
    try:
        return np.isclose(series.astype(float), float(value), rtol=rtol, atol=atol)
    except (TypeError, ValueError):
        return series == value

def get_scores(pred, target, epsilon=1e-12):
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1-pred) * target).sum()
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1_score

def get_f1score(pred, target, epsilon=1e-12):
    _, _, f1_score = get_scores(pred, target, epsilon=epsilon)
    return f1_score

class SpikeF1scoreLoss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        return 1 - get_f1score(pred, target, self.epsilon)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_epochs, rel_final_lr):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return 1
        else:
            progress = (current_epoch - num_warmup_epochs) / max(1, num_epochs - num_warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return (cosine_decay + rel_final_lr) / (1 + rel_final_lr)
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

loss_fn = SpikeF1scoreLoss()

