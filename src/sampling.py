# src/sampling.py

import argparse
import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.ansatzes import QuantumUNet
from src.utils.schedule import make_schedule, get_default_device


def parse_args():
    parser = argparse.ArgumentParser(description="Sample MNIST-like images from a trained Qiskit quantum diffusion model.")
    
    # Path & Model Config
    parser.add_argument("--checkpoint", required=True, help="Path to the Params/ directory from training.")
    parser.add_argument("--num-qubits", type=int, default=8, help="Must satisfy 2^num_qubits = 256 for MNIST (16x16).")
    parser.add_argument("--bottleneck-qubits", type=int, default=4, help="Number of qubits in the Qiskit PQC bottleneck.")
    
    # Diffusion Config
    parser.add_argument("--T", type=int, default=10, help="Number of diffusion steps.")
    parser.add_argument("--layers", type=int, nargs='+', default=[4], 
                        help="Depths for the PQC blocks (For Qiskit, the first element is used as reps).")
    
    # Sampling Config
    parser.add_argument("--num-samples", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size used (for compatibility).")
    
    # Schedule Config
    parser.add_argument("--beta0", type=float, default=1e-2)
    parser.add_argument("--betaT", type=float, default=0.02)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--schedule-exponent", type=float, default=0.5)
    parser.add_argument("--init-variance", type=float, default=0.7)
    
    # Output
    parser.add_argument("--output-dir", default="Images", help="Directory to save sample grids.")
    
    # Legacy / Unused (but kept for argument compatibility if needed)
    parser.add_argument("--noise-factor", type=float, default=0.0)
    parser.add_argument("--activation", action="store_true", help="Enable activation (Optional).")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Device Setup
    device = get_default_device()
    print(f"Using device: {device}")

    # 2. Schedule Setup
    betas, _ = make_schedule(args.beta0, args.betaT, args.T, args.schedule, args.schedule_exponent, device)

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")

    # 3. Model Initialization (Qiskit QuantumUNet)
    print("Initializing Qiskit-based QuantumUNet...")
    
    pqc_layers = args.layers[0] if isinstance(args.layers, list) else args.layers

    circuit = QuantumUNet(
        num_qubits=args.num_qubits,
        layers=args.layers,           
        T=args.T,
        init_variance=args.init_variance,
        betas=betas,
        activation=args.activation,
        device=device,
        bottleneck_qubits=args.bottleneck_qubits, # bottleneck
        use_pooling=False             
    ).to(device)

    # 4. Load Parameters
    print(f"Loading parameters from {args.checkpoint}...")
    try:
        # noise 인자는 현재 Qiskit 구현체에서 사용되지 않더라도 호환성을 위해 남겨둠
        circuit.load_best_params(args.checkpoint, noise=args.noise_factor)
        print("Loaded best parameters.")
    except FileNotFoundError:
        print("Best params not found, trying current params...")
        try:
            circuit.load_current_params(args.checkpoint, noise=args.noise_factor)
            print("Loaded current parameters.")
        except FileNotFoundError:
            raise FileNotFoundError("No model checkpoints found.")

    circuit.eval()

    # 5. Prepare Sampling
    dim = 2 ** args.num_qubits
    side = int(math.isqrt(dim))
    if side * side != dim:
        raise ValueError(f"Cannot reshape state of dimension {dim} into a square image.")

    os.makedirs(args.output_dir, exist_ok=True)

    final_outputs = np.zeros((args.num_samples, side, side))
    histories = np.zeros((args.num_samples, args.T, side, side))

    # 6. Sampling Loop (Reverse Diffusion)
    print(f"Starting sampling of {args.num_samples} images...")
    
    for sample_idx in tqdm(range(args.num_samples), desc="Sampling"):
        with torch.no_grad():
            # Start from pure noise: Complex normal distribution
            # Shape: (1, dim, 2) -> Complex Tensor
            batch = torch.randn(1, dim, 2, device=device)
            batch = torch.view_as_complex(batch).to(torch.complex64)
            
            # Normalize initial noise
            batch = batch / (torch.norm(batch, p=2, dim=1, keepdim=True) + 1e-8)

            batch_history = torch.zeros(args.T, 1, dim, dtype=torch.complex64, device=device)

            # Iterate backwards: T-1 -> 0
            for t in range(args.T - 1, -1, -1):
                # Construct input for the model: (T, Batch, Dim)
                large_batch = torch.zeros(args.T, 1, dim, device=device, dtype=torch.complex64)
                large_batch[t] = batch

                # Forward pass
                output_large_batch = circuit(large_batch)
                
                # Extract output at time t
                batch = output_large_batch[t, :, :]
                
                # Renormalize (Quantum States must be normalized)
                batch = batch / (torch.norm(batch, p=2, dim=1, keepdim=True) + 1e-8)
                
                batch_history[t] = batch.unsqueeze(0)

            # 7. Post-processing (Magnitude for visualization)
            final_output = torch.abs(batch).squeeze().cpu().numpy().reshape(side, side)
            final_outputs[sample_idx] = final_output

            batch_history_np = torch.abs(batch_history).squeeze().cpu().numpy().reshape(args.T, side, side)
            histories[sample_idx] = batch_history_np

    # 8. Visualization
    print("Saving images...")
    
    # Save Final Grid
    grid_side = math.ceil(math.sqrt(args.num_samples))
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(grid_side * 2, grid_side * 2))
    axes_flat = np.array(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i < args.num_samples:
            ax.imshow(final_outputs[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "Quantum_Generated_Samples.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Sample grid saved to {save_path}")

    # Save Diffusion History (First 5 samples only to save space)
    num_history_samples = min(5, args.num_samples)
    fig, axes = plt.subplots(num_history_samples, args.T, figsize=(args.T * 1.5, num_history_samples * 1.5))
    # Handle case where num_samples or T is 1 to ensure axes is 2D array
    if num_history_samples == 1 and args.T == 1:
        axes_grid = np.array([[axes]])
    elif num_history_samples == 1:
        axes_grid = np.array(axes).reshape(1, -1)
    elif args.T == 1:
        axes_grid = np.array(axes).reshape(-1, 1)
    else:
        axes_grid = np.array(axes)

    for i in range(num_history_samples):
        for t in range(args.T):
            axes_grid[i, t].imshow(histories[i, t], cmap='gray')
            axes_grid[i, t].axis('off')
            if i == 0:
                axes_grid[i, t].set_title(f"t={t}")
                
    plt.tight_layout()
    history_path = os.path.join(args.output_dir, "Quantum_Diffusion_Process.png")
    plt.savefig(history_path)
    plt.close(fig)
    print(f"Diffusion process history saved to {history_path}")


if __name__ == "__main__":
    main()