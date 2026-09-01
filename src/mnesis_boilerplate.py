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
# RECOMPUTE = True 
DEBUG = 1 # production
if DEBUG > 1:
    print(f'running in debug mode with DEBUG = {DEBUG}')

datetag = '2026-07-11' # run with new parameters from the camera ready
datetag = '2026-08-06' # novel run on the revamped code
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
    """Hyperparameters for the MNESIS spiking-neural network experiments."""
    datetag: str = datetag  
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
    lif_threshold: float = 0.72
    learn_beta: bool = False
    learn_threshold: bool = False
    do_pinv: bool = True
    do_deconv: bool = True

    # learning
    num_epochs: int = 256 // DEBUG
    num_warmup_epochs: int = 16          # 2**4
    base_lr: float = 30.0e-3
    final_lr: float = 1.e-3
    delta1: float = 10.e-3
    delta2: float = 10.e-6
    dropout: float = 0.25
    alpha_surrogate: float = 5.0
    surrogate_name: str = "FastSigmoid"
    loss_name: str = "SpikeF1scoreLoss"  # 'MSELoss' #'L1Loss'
    reset_mechanism: str = "subtract"    # "zero"
    optimizer: str = "sgd"              # 'adamw' #adam

    # figures
    verbose: bool = False                # Displays more verbose output.
    fig_width: float = 30                # width of figure
    fig_height: float = 15                # width of figure
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

# plt.style.use(['nature', 'science', 'prl'])   # or 'nature' for a colour figure

# --- Constants ---
phi = np.sqrt(5)/2 + 1/2
subplotpars = SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)
i_pattern = 0

# --- Utility Functions ---
def pprint(s):
    print(len(s)*'=')
    print(s)
    print(len(s)*'=')

def printfig(fig, name='', fig_width=12, fig_height=None, exts=['pdf', 'png', 'svg'], figpath=figpath, dpi_exp=None, bbox='tight', verbose=True, do_overwrite=False):
    if fig_height is None: fig_height = fig_width/phi
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches((fig_width*cm, fig_height*cm))
    if figpath is not None: 
        figpath.mkdir(exist_ok=True)
        for ext in exts:
            filename = figpath / f'{name}.{ext}'
            if filename.exists() and not do_overwrite:
                if verbose: print(f'File {filename} already exists. Skipping save.')
            else:
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
        print(f"Flipping {mask.sum().item()} bits out of {a.numel()} (p_flip={p_flip}, flip seed={seed}), {a.mean().item():.3e} -> {torch.where(mask == 1., 1 - a, a).mean().item():.3e}")
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
    """
    
    High precision → few false positives (FP, Predicted Positive is actually Negative)
    High recall → few false negatives (FN, Predicted Negative is actually Positive)
   
    """
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1-pred) * target).sum()
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1_score

def get_f1score(pred, target, epsilon=1e-12):
    """
    
    The F1 score is the harmonic mean of precision and recall, is high only when both precision and recall are high.
    
    """
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
