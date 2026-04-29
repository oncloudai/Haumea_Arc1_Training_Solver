---
license: mit
task_categories:
- question-answering
- text-generation
- other
tags:
- arc
- logic
- program-synthesis
- reasoning
- abstraction
pretty_name: Haumea ARC-1 Programmatic Reasoning
size_categories:
- n<1K
---

# Haumea ARC-1 Training Solver

A modular, law-based solver for the Abstraction and Reasoning Corpus (ARC-1) dataset. This solver successfully solves **400/400** training tasks from the ARC-1 dataset using a systematic composition of geometric, topological, and logical "laws."

## Overview

The solver is structured around a "Mega Engine" that applies a library of modular laws to solve complex visual reasoning tasks.

- **Solve Rate:** 400/400 (ARC-1 Training Set)
- **Methodology:** Modular Law Composition
- **Key Categories:** Geometric, Tiling, Motion, Recoloring, Symmetry, Logic, Filling, Stamping, and Topology.

## Directory Structure

```text
solver/
├── engine.py       # Core execution engine
├── main.py         # Entry point and law registration
├── utils.py        # Vision and grid processing utilities
└── laws/           # Categorized modular solvers
    ├── geometric/
    ├── tiling/
    ├── motion/
    ├── recoloring/
    ├── symmetry/
    ├── logic/
    ├── filling/
    ├── stamping/
    └── topology/
```

## Background

This solver is based on the **Abstraction and Reasoning Corpus (ARC)** introduced by François Chollet:

> Chollet, F. (2019). [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547). arXiv preprint arXiv:1911.01547.

## Note on Generalization & LLM Training

This release provides a high-performance modular engine for the ARC-1 training set. It is important to note:
- **Generalization:** Further work is required to achieve broad generalization across unseen tasks. This specific implementation focuses on the systematic composition of laws for a fixed dataset.
- **LLM Training:** This codebase is particularly useful as a high-quality dataset of programmatic "laws" and reasoning steps, which can be used to fine-tune or train Large Language Models (LLMs) on structured reasoning and grid-based logic.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oncloudai/Haumea_Arc1_Training_Solver.git
   cd Haumea_Arc1_Training_Solver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the solver on the ARC-1 training set:

```bash
python -m solver.main
```

## Citation

If you find this work useful in your research, please cite it as:

```text
Gullapalli, C., & Tadimati, M. H. (2026). Haumea ARC-1 Training Solver (v1.0.2). Zenodo. https://doi.org/10.5281/zenodo.19782111
```

For more details, see the [CITATION.cff](CITATION.cff) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
