#!/bin/bash

#SBATCH --job-name=trainer_1       # Name of the job
#SBATCH --output=trainer_output_no_measure.log  # Standard output log file
#SBATCH --error=trainer_error_no_measure.log    # Standard error log file
#SBATCH --mem=16G                    # Allocate 16 GB of memory
#SBATCH --gres=gpu:1                 # Request 1 GPU

srun --export=ALL python3 -m src.trainer
