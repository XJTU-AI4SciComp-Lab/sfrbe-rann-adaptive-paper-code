# SFRBE-RaNN paper code

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

MATLAB implementation of the paper
**"Adaptive Randomized Neural Networks for PDEs: Structure-Aware Feature
Redistribution and Residual-Driven Block Enrichment"**.

The hidden random features are generated once and kept fixed during each
solve; only the output-layer coefficients are determined, from a linear or
linearized least-squares system. SFRBE-RaNN adapts such a trial space in two
stages, SFR and RBE.

## Paper Information

- Paper title: Adaptive Randomized Neural Networks for PDEs: Structure-Aware Feature Redistribution and Residual-Driven Block Enrichment
- Authors: You Yang, Fei Wang
- Preprint: arXiv:2610.21077 [math.NA], cross-list cs.LG (15 October 2026)
- Journal: submitted for publication (under review)
- DOI / arXiv: https://arxiv.org/abs/2610.21077
- Corresponding author: Fei Wang (feiwang.xjtu@xjtu.edu.cn)
- Code maintainer: You Yang (@Onlooker-DL)
- Repository: https://github.com/XJTU-AI4SciComp-Lab/sfrbe-rann-adaptive-paper-code

## Method Summary

- **SFR (structure-aware feature redistribution)** keeps the total feature
  budget fixed. Structure regions are detected from residual, derivative,
  geometric, or problem-specific indicators; the budget is redistributed
  among the detected regions and the background according to their relative
  indicator masses, and region-wise random-weight scales are adapted by
  low-dimensional residual-based optimization.
- **RBE (residual-driven block enrichment)** expands the trial space with
  nonredundant blocks. Candidate blocks from several activation families are
  residualized against the current response space, ranked by a common capture
  score, and truncated at the smallest admissible prefix. For
  solution-dependent operators, candidates are screened with a frozen
  operator and accepted only after nonlinear residual verification; fixed
  linear problems use block-incremental residual updates.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── startup.m                     add src/ to the MATLAB path
├── src/                          algorithm library
│   ├── basis/                      basis functions, operator matrices, RBE evaluation
│   ├── train/                      AD-RaNN and PDAD scale training
│   ├── sfr/                        SFR module
│   ├── rbe/                        RBE family screening and block selection
│   ├── solver/                     fixed-operator and Newton solvers
│   ├── utils/                      sampling, SVD initialization, error metrics
│   ├── io/                         result saving, run logs, paper figure export
│   ├── problem/                    shared problem helpers (Darcy phi field)
│   └── burgers3d/                  space-time Burgers pipeline modules
├── problems/                     one ProblemFunction per PDE
├── experiments/                  one folder per paper experiment
│   └── run_experiment.m            shared pipeline driver
├── examples/                     small end-to-end example
├── data/                         reference grids and reference solutions
├── results/                      generated numerical results
├── figures/                      exported paper figures
├── tests/                        additional tests
└── docs/                         additional documentation
```

## Installation

Requirements:

- MATLAB (base MATLAB only)
- Parallel Computing Toolbox (`gather` is used and `gpuArray` is optional; a
  GPU is not required by the default configurations)

No third-party MATLAB packages are required. Clone the repository and start
MATLAB in the repository root:

```bash
git clone https://github.com/XJTU-AI4SciComp-Lab/sfrbe-rann-adaptive-paper-code.git
cd sfrbe-rann-adaptive-paper-code
```

`requirements.txt` records these MATLAB requirements; there are no Python
dependencies and nothing to install with pip. `environment.yml` is kept for lab
repository consistency and is not used by the MATLAB code.

## Quick Start

Minimal end-to-end run of the full pipeline (PDAD -> SFR -> RBE -> evaluation)
on a coarse 60x60 grid with 600 initial features and one RBE round, about one
minute:

```matlab
startup
cd examples
example_poisson_2peak
```

The output is written to `results/example_poisson_2peak/` and does not
overwrite the experiment results used in the paper.

## Run an Experiment

Nine of the folders under `experiments/` follow one convention: a `config.m`
holding all hyperparameters and the default seeds, and a `main.m` entry point.
Two folders differ. `poisson_2peak_seed4_detailed/` has no `config.m` of its
own; it reuses `experiments/poisson_2peak/config.m` and overrides only the
seed, the run name and the number of RBE rounds. `burgers3d/` is a standalone
pipeline with `burgers3d_config.m` and `run_burgers3d.m`. For example:

```matlab
cd experiments/poisson_2peak
main
```

`main` accepts either a scalar seed or a vector of seeds. With a vector, one
run is performed per seed and a mean/standard-deviation summary is printed and
saved; the standalone `run_burgers3d` takes a single seed from
`burgers3d_config.m`. Default seeds:

| Experiment | Default seed(s) |
| --- | --- |
| `poisson_2peak`, `poisson_2peak_ad`, `poisson_2peak_sfr`, `poisson_2peak_rbe_only` | `0:9` |
| `kdv62_seeds` | `0:9` |
| `poisson_2peak_seed4_detailed`, `helmholtz`, `darcy/phi`, `kdv62`, `kdv_general_fixed`, `burgers3d` | `4` |

To customize a single run, edit `config.m`, or call the driver directly with a
scalar seed:

```matlab
startup
addpath('experiments')
addpath('experiments/poisson_2peak')

cfg = config();
cfg.seed = 4;
cfg.grid.N = [100, 100];
cfg.output.run_name = 'poisson_2peak_custom';
run_experiment(cfg);
```

Results are written to `results/<run_name>/`, or to
`results/<problem>_seed<k>/` for a multi-seed sweep.

The two-dimensional-in-space, three-dimensional space-time Burgers pipeline is
standalone and uses the fixed-operator Picard path by default:

```matlab
cd experiments/burgers3d
run_burgers3d
```

## Smoke Test

A fast check of the toolchain (about one minute, coarse 60x60 grid, seed 1):

```matlab
startup
cd experiments/poisson_2peak
smoke
```

The smoke run writes to `results/poisson_2peak_smoke/` and does not overwrite
the paper results.

## Reproduce Results

Each experiment folder reproduces one block of the paper; the figure and table
numbers refer to the submitted manuscript.

| Folder | Reproduces |
| --- | --- |
| `experiments/poisson_2peak` | two-peak Poisson: Table 2, Table 3, Figure 2, Figure 3 |
| `experiments/poisson_2peak_ad` | AD-RaNN baseline row of Table 4 |
| `experiments/poisson_2peak_sfr` | SFR-RaNN ablation row of Table 4 |
| `experiments/poisson_2peak_rbe_only` | RBE-RaNN (no SFR) ablation row of Table 4 |
| `experiments/poisson_2peak_seed4_detailed` | seed-4 diagnostics of Table 5 and the RBE log of Table 13 |
| `experiments/helmholtz` | multiscale Helmholtz: Figure 5, Table 6 |
| `experiments/darcy/phi` | smooth multiscale Darcy: Figure 6, Table 7 |
| `experiments/kdv_general_fixed` | generalized KdV, fixed operator: Figure 7, Table 8 |
| `experiments/kdv62`, `experiments/kdv62_seeds` | periodic KdV two-soliton: Figure 8, Table 9 |
| `experiments/burgers3d` | two-dimensional-in-space Burgers: Figure 9, Table 10 |

Table 1 (activation families), Table 11 and Table 12 (problem and solver
settings) are configuration summaries rather than numerical output, and
Figure 1 is the method schematic. Figure 4 compares with adaptive finite
elements; that adaptive FEM reference code is not part of this repository.

Paper figures are exported as compressed PDF and 300 dpi PNG when a run has
`cfg.output.make_figs = true`; this is enabled in the main Poisson, Darcy,
KdV62 and generalized KdV configurations and disabled in the baseline and
ablation configurations, whose figures can be produced afterwards from saved
results with `src/io/plotFromResults.m`.

## Folder Description

- `src/`: algorithm library (basis, training, SFR, RBE, solvers, utilities, I/O, Burgers modules).
- `problems/`: PDE definitions, one `ProblemFunction.m` per problem (`poisson_2peak`, `lshape`, `helmholtz`, `darcy_phi`, `kdv_general`, `kdv62`). `lshape` is a problem definition without a dedicated experiment folder and is not used in the paper tables or figures.
- `experiments/`: one folder per paper experiment, plus the shared driver `run_experiment.m`.
- `examples/`: small runnable example.
- `data/`: reference grids and reference solutions; see `data/README.md`.
- `results/`: generated numerical results.
- `figures/`: exported paper figures, in `figures/<problem>/`.
- `tests/`: additional tests; the current smoke test lives in `experiments/poisson_2peak/smoke.m`.
- `docs/`: additional documentation.

## Data

The reference inputs are small and are tracked in the repository:

- `data/darcy_phi.mat`: reference grid and solution for the smooth multiscale
  Darcy experiment (`x` 101x1, `y` 101x1, `u` 101x101).
- `data/KdV62_reference_100x100.mat`: high-accuracy Fourier spectral reference
  for the periodic KdV two-soliton experiment (`x_ref` 100x1, `t_ref` 100x1,
  `u_ref` 100x100).

No large dataset is needed. If an experiment requires a large file, keep it
outside the repository and document the download link or the generation script
in `data/README.md`.

## Results

Generated results are written to `results/`; result data files and logs are
excluded by `.gitignore`. Each run writes a compact `run.log`, plotting
snapshots and a `99_summary.mat`, and a multi-seed sweep additionally writes
`<problem>_sweep.txt` in `results/`. Exported figures go to
`figures/<problem>/`. Wall-clock times reported by the code are machine
dependent.

## Reproducibility Notes

- All random stages use `cfg.seed`, and the standard entry points run one
  pipeline call per scalar seed.
- All hyperparameters live in each experiment's `config.m`.
- The code contains no absolute paths: data and results are located relative
  to the repository root through `mfilename('fullpath')`.
- Training stages retain the iterate with the smallest residual mean-square
  error.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{yang2026adaptiverandomizedneuralnetworks,
  title        = {Adaptive Randomized Neural Networks for {PDEs}: Structure-Aware Feature Redistribution and Residual-Driven Block Enrichment},
  author       = {You Yang and Fei Wang},
  year         = {2026},
  eprint       = {2610.21077},
  archivePrefix = {arXiv},
  primaryClass = {math.NA},
  url          = {https://arxiv.org/abs/2610.21077}
}
```

The same information is available in `CITATION.cff`; the journal reference and
DOI will be added once the paper is published.

## License

This project is released under the BSD 3-Clause License; see `LICENSE`.

## Contact

Fei Wang
School of Mathematics and Statistics, Xi'an Jiaotong University, Xi'an, Shaanxi 710049, China.
Email: feiwang.xjtu@xjtu.edu.cn
