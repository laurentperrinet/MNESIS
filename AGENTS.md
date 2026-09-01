# AGENTS.md — Developer & AI-Assistant Reference

## Project Summary

MNESIS is a recurrent spiking neural network (SNN) with heterogeneous synaptic
delays. The repository has two independent, buildable halves:

- **`src/`** — Python source and Jupyter experiment notebooks. Core code lives in
  two modules (`mnesis_boilerplate.py`, `mnesis_chains.py`); the numbered
  notebooks import from them and are run sequentially.
- **`tex/`** — LaTeX source of the paper (`Perrinet26mnesis.tex`), a small set of
  standalone TikZ figures, a bibliography, and the Springer LNCS class/style files.

This file gives **rules, commands, and gotchas** for agents working in the
repository; it is not a place for reference data. The full parameter catalogue,
per-experiment results, and notebook inventory live in `README.md` — point there
for "what value / what result", and to the paper (`tex/Perrinet26mnesis.pdf`) for
the camera-ready numbers.

An AI assistant was used to improve the readability of the code, not to create
it; these rules guide how that formatting and testing was done.

---

## Architecture Overview

### Module Dependency Graph (`src/`)

```
notebooks (*.ipynb)
   └── mnesis_chains.py         (HD_SNN, SpikingPattern, StochasticSpikingPattern, load())
         └── mnesis_boilerplate.py    (imports, Params dataclass, utilities, SpikeF1scoreLoss)
               └── snntorch / torch / numpy / matplotlib / ...
```

| File | Role |
|---|---|
| `src/mnesis_boilerplate.py` | Entry-point imports (torch, snntorch, …), device auto-detection (MPS → CUDA → CPU), `Params` dataclass (single source of hyperparameter truth), utilities (`flip_bits`, `printfig`, `get_scores`, `SpikeF1scoreLoss`, cosine LR schedule). |
| `src/mnesis_chains.py` | Pattern generators (`SpikingPattern`, `StochasticSpikingPattern`) and the `HD_SNN` class (`forward_pass`, `get_W_init`, `update_weight`, `learn_model`); also a `load()` helper for trained checkpoints. |

Notebooks (`src/*.ipynb`) are numbered simple → complex, each loading cached
artifacts from the previous one. They **import** from the two modules — they do
**not** carry standalone copies of `Params`, `HD_SNN`, etc. Notebooks 01, 05, and
08 (former `boilerplate`, `parameters`, `generative-model`) were refactored into
the two modules; any reference to those notebook numbers in old documents is stale.

---

## `src/` — Python source code

### Install

```bash
pip install -r src/requirements.txt     # Python 3.10+; toolchain lives in .venv/
```

### Conventions

- Imports belong at the top of `mnesis_boilerplate.py`; downstream modules and
  notebooks re-export via `from mnesis_boilerplate import …`.
- Notebooks `from mnesis_boilerplate import …` / `from mnesis_chains import …`, so
  they must be **run from inside `src/`** (that is the expected CWD).
- `src/mnesis_boilerplate.py` `Params` is the authoritative hyperparameter source.
- `.venv/` is git-ignored; do not commit it.
- Figure/sandbox output goes to `../figures/` (relative to `src/`); on the Jean Zay
  cluster (`USER == "uvb28bo"`) figure saving is disabled automatically.
- `datetag` controls the cache-filename prefix — change it to start a new batch.
- `seed = 2018` is fixed for reproducibility.

### Debug mode

Set `DEBUG > 1` in `mnesis_boilerplate.py` to shrink the problem: `N_neuron`,
`N_pattern`, `N_time`, and `num_epochs` are divided by `DEBUG`. Use `DEBUG = 1` (the
default) only for production/camera-ready runs.

### Cache invalidation

Artifacts are cached in `cached_data/` (git-ignored). To recompute:
- delete the individual `.pth`, `.json`, `.npz`, or `.sqlite3` file, or
- set `RECOMPUTE = True` at the top of the notebook to invalidate the whole cache.

**File-based locking** (used by the scan/optimisation notebooks): a `.lock`
sentinel file guards against concurrent runs of the same scan.
1. If `RECOMPUTE=True`, delete both the data and lock files to force a restart.
2. Load existing results if present.
3. If no lock file exists, create one and start processing.
4. For each parameter value, skip it if already computed; otherwise run the scan and append to the JSON.
5. Delete the lock file when done.
This avoids data corruption and allows safe incremental resumption — never delete
a `.lock` out from under a running scan.

### Test / compile the Python side

There is **no unit-test framework** (no pytest/CI). The checks below *are* the
tests; run the fast one first, then a representative notebook, and only the full
orchestrator when you need end-to-end verification.

```bash
cd src

# 1. Fast: modules import and build cleanly
python -c "import mnesis_boilerplate, mnesis_chains"

# 2. Representative: execute the smallest notebook (set DEBUG > 1 in mnesis_boilerplate.py first)
jupyter nbconvert --to notebook --execute --inplace 10_MNESIS_generative-model.ipynb

# 3. Full pipeline (slow): the orchestrator runs the whole notebook chain
jupyter nbconvert --to notebook --execute --inplace 99_MNESIS_run-all.ipynb
```

`nbconvert` exiting without error and writing updated outputs is a green build.

---

## `tex/` — LaTeX source code

### Files

| File | Role |
|---|---|
| `Perrinet26mnesis.tex` | Main paper. `\documentclass` + `\input{metadata}` + `\bibliography{mnesis}` (style `plainnat`, not biblatex). |
| `metadata.tex` | `\input` fragment (author/affiliation/AI-statement macros); **no** `\documentclass`, do not compile standalone. |
| `fig_izhikevich.tex`, `fig_snntorch.tex` | Standalone TikZ figures (`\documentclass{standalone}`); compiled to their own PDFs in `tex/`. |
| `mnesis.bib` | Bibliography database. |

The main paper resolves experiment figures through `\graphicspath{{../figures/}}`
(`Perrinet26mnesis.tex:48`): `pattern.pdf`, `target.pdf`, `retrieval.pdf`, the
`*_score.pdf` curves, etc. are **generated by the `src/` notebooks**, not built in
`tex/`. The two TikZ figures are built in `tex/`. Build everything from inside
`tex/`. Engine is `pdflatex` (with `latexmk`); toolchain: `pdflatex`, `latexmk`,
`bibtex`/`biber`.

### Test / compile the LaTeX side

Full build (reproduces `tex/Perrinet26mnesis.pdf`):

```bash
cd tex

# 1. Standalone TikZ figures
pdflatex -interaction=nonstopmode fig_izhikevich.tex
pdflatex -interaction=nonstopmode fig_snntorch.tex

# 2. Experiment figures — must already exist in ../figures/
#    generate them by running the relevant notebooks in src/ (see above), e.g.
cd ../src && jupyter nbconvert --to notebook --execute --inplace 11_MNESIS_learn-synthetic.ipynb && cd ..

# 3. Build the paper (bibtex handled automatically by latexmk)
cd tex && latexmk -pdf -interaction=nonstopmode Perrinet26mnesis.tex
```

Quick, figure-independent parse check (ignores missing `../figures/*.pdf`):

```bash
cd tex && pdflatex -interaction=nonstopmode -halt-on-error=false Perrinet26mnesis.tex
```

A clean run produces `tex/Perrinet26mnesis.pdf`. Generated byproducts
(`.aux`, `.bbl`, `.blg`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, and the
`Perrinet26mnesis_copy.tex*` scratch copy) are git-ignored — never commit them.

---

## Conventions & gotchas

- **Parameter drift.** Some defaults in `mnesis_boilerplate.py` differ from the
  paper text and `README.md` (e.g. `lif_beta = 0.8` vs 0.7, `lif_threshold = 0.72`
  vs 1.0, `alpha_surrogate = 5.0` vs 15, `dropout = 0.25` vs 0.37). The code
  defaults are the latest camera-ready run (the current `datetag`, `2026-08-06`).
  Each notebook also overrides the defaults with its own local `opt_dict`. **Do not
  "autocorrect" code defaults to match the prose** — consult
  `README.md` / `tex/Perrinet26mnesis.pdf` for the exact value a given figure used.
- The analytical initialisation in `HD_SNN.get_W_init` is controlled by two flags:
  `do_pinv` (default `True`; `False` switches to the Hebbian cross-correlation rule)
  and `do_deconv` (default `True`; LIF-membrane deconvolution of the target).
- `num_delay` must stay **odd** (convolution symmetry).
- Keep notebooks' shared code **in the two modules**, not inlined — notebooks are
  meant to stay import-only.
