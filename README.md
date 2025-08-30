# Automated carbide detection

Tool to detect carbides in scanning electron micrographs of steel.

During the production and heat treatment of steel, carbides with a size ranging between 10 nm up to a few µm precipitate in the steel matrix. While the carbides contribute to the steels yield strength, the largest carbides can be responsible for crack initiation leading to brittle fracture. Thus, a detailed quantitative description of carbides (e.g. number density, size distribution etc.) in a steel is of great interest.
On a polished sample, carbides can be observed using a scanning electron microscope (SEM). In the present case, SEM micrographs of a reactor pressure vessel steel have been recorded, in which carbides can be recognized.

## Installation

These instructions are adapted for https://anonymous.4open.science they will be different for the code released on github

1. download the repository and unpack it
   ```bash
   mkdir carbide_detection
   cd carbide_detection
   curl -o carbide_detection.zip https://anonymous.4open.science/api/repo/carbide_detection/zip
   unzip carbide_detection.zip
   ```
2. download and extract the data as described in [data/README.md](data/README.md)
3. create a virtual environment (using python 3.11 or above), and install the dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install uv
   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CARDE=0.0.1 uv pip install -e .
   ```
   in case you would like to reproduce the hyperparameter optimization or run the unit tests, install the optional `dev` dependencies
   ```bash
   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CARDE=0.0.1 uv pip install -e ".[dev]"
   ```

## Reproducing the Paper Results

Open the notebooks and execute all cells sequentially to reproduce the results:

- [docs/notebooks/run_unet_experiments.ipynb](docs/notebooks/run_unet_experiments.ipynb)
  Generates segmentation results and plots the figures shown in the manuscript.

- [docs/notebooks/uncertainty_estimation.ipynb](docs/notebooks/uncertainty_estimation.ipynb)
  Reproduces unertainty estimation via temperature scaing.

- [docs/notebooks/hyperparameter_optimization.ipynb](docs/notebooks/hyperparameter_optimization.ipynb)
  Reproduces hyperparameter optimization experiments.
