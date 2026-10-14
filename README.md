# Paper Code Template

This repository provides a standard template for paper-related code developed by the XJTU AI for Scientific Computing Lab.

It is intended for randomized neural networks, local randomized neural networks, adaptive and growing randomized neural networks, operator learning models, and structure-preserving AI methods for scientific computing.

## Paper Information

- Paper title:
- Authors:
- Journal / preprint:
- DOI / arXiv:
- Corresponding author:
- Code maintainer:
- Repository:

## Method Summary

Briefly describe the method used in this paper.

Example:

This repository implements a randomized neural network method for solving differential equations. The hidden-layer parameters are randomly generated and fixed, while the output-layer coefficients are computed from a least-squares system.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── src/
├── examples/
├── experiments/
├── data/
├── results/
├── figures/
├── tests/
└── docs/
```

## Installation

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate rann-paper-code
```

Or install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a minimal example:

```bash
python examples/run_demo.py
```

## Reproduce Results

Use scripts in `experiments/` to reproduce figures and tables in the paper.

```bash
python experiments/reproduce_table1.py
python experiments/reproduce_figure1.py
```

## Folder Description

- `src/`: source code for the proposed method.
- `examples/`: simple runnable examples.
- `experiments/`: scripts for reproducing paper figures and tables.
- `data/`: small datasets or instructions for obtaining data.
- `results/`: generated numerical results.
- `figures/`: generated figures or plotting scripts.
- `tests/`: basic tests for important modules.
- `docs/`: additional documentation.

## Data

Small datasets may be included in `data/`.

Large datasets should not be committed directly to this repository. Instead, provide download links, data-generation scripts, or instructions in `data/README.md`.

## Results

Generated results should be saved in `results/`.

Large result files should not be committed unless they are necessary for reproducing the paper.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@article{paper_key,
  title   = {},
  author  = {},
  journal = {},
  year    = {}
}
```

## License

Please see `LICENSE`.

## Contact

Fei Wang  
School of Mathematics and Statistics, Xi'an Jiaotong University.
