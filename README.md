Based on the papers:
- [Hybrid and hardware-oriented approaches for quantum diffusion models](https://ieeexplore.ieee.org/document/11227283)
- [Quantum diffusion models](https://arxiv.org/abs/2311.15444)

# Quantum Diffusion on MNIST

MNIST-only quantum diffusion models with two architectures:
- `PQC`: three stacked parameterized circuits with an ancilla in the middle block.
- `NNCPQC`: MLP-driven activation parameters plus PQC/activation stacks.

## Setup
- Create the environment: `conda env create -f environment.yml && conda activate qdm_venv`
- Data is downloaded automatically to `data/` on first run.

## Training
- Run: `python3 -m src.trainer`
- Hyperparameters live in `src/trainer.py` (`model_type`, learning rates, depths, digits, etc.). MNIST inputs are 16x16 (flattened to 256), so keep `num_qubits=8`.
- Outputs per run go to `results/<timestamp>/run_*/trial_*/`:
  - `Params/` holds `current*.pt` and `best*.pt` checkpoints.
  - `TensorBoard/` contains scalars and sampled image grids (logged every 50 epochs).
- On clusters, `job.sbatch` runs the same entrypoint (`sbatch job.sbatch`).

## Sampling
- Requires a trained `Params/` directory.
- Example (PQC):  
  `python3 -m src.sampling --checkpoint results/<ts>/run_1/trial_1/Params --model-type PQC --layers 4 4 4`
- Example (NNCPQC):  
  `python3 -m src.sampling --checkpoint results/<ts>/run_1/trial_1/Params --model-type NNCPQC --num-layers 20 --MLP-depth 5 --MLP-width 64 --PQC-depth 4 --ACT-depth 4 --num-ancilla 1 --batch-size 64`
- Images are saved under `Images/` and intermediate denoising trajectories per sample are saved as grids.

## Visualizing Training
- TensorBoard: `tensorboard --logdir=results/` (losses, learning rates, diffusion forward pass, generated samples).
- Checkpointed samples: see `Images/` from the sampling script for qualitative outputs.
