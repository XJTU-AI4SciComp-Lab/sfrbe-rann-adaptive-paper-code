# SFRBE-RaNN

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

MATLAB implementation accompanying the paper **"Adaptive Randomized Neural
Networks for PDEs via Structure-Aware Feature Redistribution and
Residual-Driven Block Enrichment."**

The repository contains reusable randomized-neural-network solvers, problem
definitions, experiment configurations, a small example, and a smoke test.

## Requirements

- MATLAB
- Parallel Computing Toolbox (the implementation uses `gather` and can
  optionally use `gpuArray`; a GPU is not required by the default configs)

No third-party MATLAB packages are required.

## Quick start

Clone the repository, start MATLAB in the repository root, and run:

```matlab
startup
cd example
example_poisson_2peak
```

The example uses a coarse grid, 600 initial features, one RBE round, and the
scalar random seed `4`. Its output is written to
`results/example_poisson_2peak/`.

## Run an experiment

Each experiment has a `config.m` and a `main.m`. For example:

```matlab
cd experiments/poisson_2peak
main
```

`main` supports either a scalar seed or a vector of seeds. The default
Poisson two-peak configuration runs seeds 0 through 9 and produces a
mean/standard-deviation summary.

To customize a single run, make sure `cfg.seed` is a scalar before calling
`run_experiment` directly:

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

Available experiment folders are:

- `poisson_2peak`
- `poisson_2peak_ad`
- `poisson_2peak_rbe_only`
- `poisson_2peak_seed4_detailed`
- `poisson_2peak_sfr`
- `lshape`
- `helmholtz`
- `darcy/phi`
- `kdv_general_fixed`
- `kdv62`
- `kdv62_seeds`
- `burgers3d`

The `burgers3d` folder contains the standalone two-dimensional-in-space,
three-dimensional space--time Burgers pipeline used in the paper. It uses the
fixed-operator Picard path by default. Start it from MATLAB with:

```matlab
cd experiments/burgers3d
run_burgers3d
```

## Smoke test

From the repository root, run:

```matlab
startup
addpath('tests')
smoke_poisson_2peak
```

The smoke test writes to `results/poisson_2peak_smoke_test/` and does not
overwrite the full Poisson experiment results.

## Repository layout

```text
src/           reusable algorithm library
  basis/       basis functions and operator matrices
  train/       AD-RaNN and PDAD scale training
  sfr/         structure-aware feature redistribution (SFR)
  rbe/         residual-driven block enrichment (RBE)
  solver/      Newton solver
  utils/       sampling, least-squares, and SVD utilities
  io/          result saving and figure export
problems/      PDE definitions (one folder per problem)
experiments/   experiment configs and entry points
example/       small end-to-end example
tests/         smoke tests
data/          optional reference data
results/       generated numerical results (ignored by Git)
figures/       generated paper figures
```

## Reproducibility and outputs

- Random stages use `cfg.seed`; the standard experiment entry points handle
  multi-seed sweeps by invoking the pipeline once per scalar seed.
- Hyperparameters are stored in each experiment's `config.m`.
- The training stages retain the iterate with the smallest residual
  mean-square error.
- Each run writes a compact `run.log`, plotting snapshots, and a
  `99_summary.mat` file under `results/`.
- Generated results, logs, figures, and local backup files are
  excluded by `.gitignore`.

The KdV62 experiment uses the tracked reference file
`data/KdV62_reference_100x100.mat`. The smooth multiscale Darcy experiment
uses the tracked reference grid `data/darcy_phi.mat`.

## Notes for GitHub publication

The repository uses the BSD 3-Clause License. Generated results and figures
are kept local and are not included in the source distribution. If the work
is associated with a paper, add its final citation and DOI when available.
