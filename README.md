# Drain-Vortex Optimization (DVO)

This repository contains the implementation and experimental scripts for **Drain-Vortex Optimization (DVO)**, a physics-inspired population-based metaheuristic algorithm based on radial and tangential drain-vortex dynamics.

The repository accompanies the manuscript:

> **Drain-Vortex Optimization: A Physics-Inspired Metaheuristic Based on Radial and Tangential Drain-Vortex Dynamics**

## Overview

DVO is inspired by the flow structure of a drain vortex. The algorithm models candidate solutions as agents moving under:

- radial attraction toward drain centres,
- free-vortex tangential motion,
- adaptive spiral exploitation,
- population-level assignment to multiple drain basins,
- optional stochastic transfer between drain basins for diversity.

The implementation also includes several baseline optimizers used in the paper, including PSO, GWO, WOA, SCA, AOA, EO, and SVOA.

## Repository structure

```text
.
├── algorithms/              # DVO and baseline optimizer implementations
├── experiments/             # Benchmark and analysis scripts
├── paper_tables/            # LaTeX tables generated from the experiments
├── results_with_svoa/        # Processed results used in the manuscript
├── requirements.txt         # Python dependencies
├── README.md
└── LICENSE
```

## Installation

The code was tested with Python 3.10 and PyTorch.

Create a clean environment:

```bash
conda create -n dvo python=3.10 -y
conda activate dvo
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build that matches your CUDA version.

## Running a quick smoke test

From the repository root:

```bash
python experiments/run_cec2022_comparison.py \
  --mode smoke \
  --algorithms MVDO PSO GWO WOA SCA AOA EO SVOA
```

Note: in the codebase, the proposed method may still appear as `MVDO` for historical reasons. In the manuscript, it is reported as **DVO**.

## Running the main experiments

### CEC2022

```bash
python experiments/run_cec2022_comparison.py \
  --mode full \
  --algorithms MVDO PSO GWO WOA SCA AOA EO SVOA
```

### CEC2017

```bash
python experiments/run_cec2017_comparison.py \
  --mode full \
  --algorithms MVDO PSO GWO WOA SCA AOA EO SVOA
```

### Classical benchmark functions

```bash
python experiments/run_classical_comparison.py \
  --mode full \
  --algorithms MVDO PSO GWO WOA SCA AOA EO SVOA
```

### Engineering design problems

```bash
python experiments/run_engineering_comparison.py \
  --mode full \
  --algorithms MVDO PSO GWO WOA SCA AOA EO SVOA
```

### Ablation study

```bash
python experiments/run_ablation_cec2017_subset.py --mode full
```

### Convergence and stability analysis

```bash
python experiments/run_cec2017_convergence_subset.py \
  --algorithms MVDO PSO GWO EO SVOA
```

## Reproducing the paper tables

After running the experiments, the LaTeX tables can be regenerated using:

```bash
python experiments/make_latex_tables_with_svoa.py
```

The generated tables are saved in:

```text
paper_tables_with_svoa/
```

## Benchmark data

Some CEC benchmark data files may not be redistributed in this repository. If a benchmark script requires external data, please download the corresponding official benchmark data and place it in the expected data directory.

Recommended structure:

```text
data/
├── cec2022_data/
└── cec2017_data/
```

If needed, update the data path in the corresponding experiment script or pass the path through the script arguments.

## Results

The processed result files used for the manuscript are included in:

```text
results_with_svoa/
```

These files include the merged results after adding SVOA to the comparison.

## Citation

If you use this code, please cite the associated preprint:

```bibtex
@article{omidi2026dvo,
  title   = {Drain-Vortex Optimization: A Physics-Inspired Metaheuristic Based on Radial and Tangential Drain-Vortex Dynamics},
  author  = {Omidi, Mohsen and Vaughan, Brian},
  year    = {2026},
  note    = {Preprint},
  doi ={https://doi.org/10.48550/arXiv.2605.08883
}
```

The BibTeX entry will be updated after the arXiv version becomes available.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Contact

For questions, please contact:

**Mohsen Omidi**  
Technological University Dublin
