# src.trainer.py

import os
import itertools
import sys
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils.circuit_training import training

# ==============================================================================
# Hyperparameters Configuration
# ==============================================================================
hyperparameters = {
    # --- Model Architecture ---
    "model_type":        ["UNet"],       # identifier
    "num_qubits":        [8],            # Input dimension: 2^8 = 256 (16x16 image)
    "bottleneck_qubits": [4],            # Number of Qiskit PQC
    "pqc_layers":        [6],            # Reps of Qiskit RealAmplitudes
    "activation":        [False],        # Activation

    # --- Diffusion Process ---
    "T":                 [10],           # Diffusion Time Steps
    "beta_0":            [0.0001],       # Noise schedule start
    "beta_T":            [0.02],         # Noise schedule end
    "schedule":          ['linear'],     # 'linear' or 'cosine'
    "schedule_exponent": [0.5],          # cosine schedule exponent
    
    # --- Training Configuration ---
    "batch_size":        [32],           
    "num_epochs":        [100],           
    "PQC_LR":            [1e-3],         # Learning Rate
    "wd_PQC":            [0.0],          # L2 Regularization (Weight Decay)
    "init_variance":     [0.1],          # Parameter initialization variance
    
    # --- Scheduler ---
    "scheduler_patience":[5],            # LR Scheduler patience
    "scheduler_gamma":   [0.5],          # LR Reduction factor

    # --- Data & Misc ---
    "digits":            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], # dataset
    "checkpoint":        [None],         # Resume training from path (e.g., '/path/to/Params/')
}

DATA_LENGTH = 8096 # Number of samples to use per epoch
NUM_TRIALS = 1     # Trials per configuration
results_dir = 'results'

# ================> DON'T MODIFY BELOW THIS LINE <================

# Create path to save weights and results
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(os.getcwd(), results_dir, timestr)

# Generate all combinations of hyperparameters
keys = hyperparameters.keys()
values_list = hyperparameters.values()
all_combinations = list(itertools.product(*values_list))

print(f"Total configurations to run: {len(all_combinations)}")

# Loop over all possible combinations
for i, values in enumerate(all_combinations, start=1):
    for trial in range(1, NUM_TRIALS + 1):
        
        # 1. Config Dictionary
        config = dict(zip(keys, values))
        
        # 2. Path Setting
        run_name = f"run_{i}_trial_{trial}"
        path = os.path.join(log_dir, run_name)
        os.makedirs(path, exist_ok=True)

        print(f"\n[Start Training] {run_name}")
        print(f"Config: {config}")

        # 3. training
        training(path, config, DATA_LENGTH)