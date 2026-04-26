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

If you find this work useful in your research, please cite it using the following metadata:

```text
[INSERT_CITATION_HERE_FROM_ZENODO_OR_CITATION_CFF]
```

For more details, see the [CITATION.cff](CITATION.cff) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
