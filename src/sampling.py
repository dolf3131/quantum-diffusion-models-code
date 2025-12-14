# src.sampling.py

import argparse
import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.ansatzes import PQC, NNCPQC, QuantumUNet
from src.utils.schedule import make_schedule


def parse_args():
    parser = argparse.ArgumentParser(description="Sample MNIST-like images from a trained quantum diffusion model.")
    parser.add_argument("--checkpoint", required=True, help="Path to the Params/ directory from training.")
    parser.add_argument("--model-type", choices=["PQC", "NNCPQC"], default="PQC")
    parser.add_argument("--num-qubits", type=int, default=8, help="Must satisfy 2^num_qubits = 256 for MNIST (16x16).")
    parser.add_argument("--T", type=int, default=10, help="Number of diffusion steps.")
    parser.add_argument("--layers", type=int, nargs=3, default=[4, 4, 4], metavar=("L1", "L2", "L3"),
                        help="Depths for the three PQC blocks (PQC only).")
    parser.add_argument("--num-layers", type=int, default=20, help="Number of unitary+activation blocks (NNCPQC).")
    parser.add_argument("--MLP-depth", type=int, default=5, help="Depth of the MLP (NNCPQC).")
    parser.add_argument("--MLP-width", type=int, default=64, help="Width of the MLP (NNCPQC).")
    parser.add_argument("--PQC-depth", type=int, default=4, help="Depth of each PQC block (NNCPQC).")
    parser.add_argument("--ACT-depth", type=int, default=4, help="Depth of each activation block (NNCPQC).")
    parser.add_argument("--num-ancilla", type=int, default=1, help="Number of ancilla qubits (NNCPQC).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size used during training (NNCPQC).")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--noise-factor", type=float, default=0.0,
                        help="Gaussian noise added to parameters at load time (e.g., 0.005).")
    parser.add_argument("--beta0", type=float, default=1e-2)
    parser.add_argument("--betaT", type=float, default=1e-2)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--schedule-exponent", type=float, default=0.5)
    parser.add_argument("--init-variance", type=float, default=0.7)
    parser.add_argument("--activation", action="store_true", help="Enable activation in the middle PQC block (PQC).")
    parser.add_argument("--output-dir", default="Images", help="Directory to save sample grids.")
    return parser.parse_args()


def main():
    args = parse_args()

    from src.utils.schedule import get_default_device
    device = get_default_device()
    betas, _ = make_schedule(args.beta0, args.betaT, args.T, args.schedule, args.schedule_exponent, device)

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")

    model_type = args.model_type.lower()
    if model_type == "pqc":
        # circuit = PQC(
        #     args.num_qubits, args.layers, args.T, args.init_variance, betas,
        #     activation=args.activation, device=device
        # ).to(device)
        circuit = QuantumUNet(
            args.num_qubits, args.layers, args.T, args.init_variance, betas,
            activation=args.activation, device=device, bottleneck_qubits=args.bottleneck_qubits
        ).to(device)
        try:
            circuit.load_best_params(args.checkpoint, noise=args.noise_factor)
        except FileNotFoundError:
            circuit.load_current_params(args.checkpoint, noise=args.noise_factor)
    else:
        circuit = NNCPQC(
            args.num_qubits, args.num_ancilla, args.num_layers, args.MLP_depth, args.MLP_width,
            args.PQC_depth, args.ACT_depth, args.T, args.init_variance, args.batch_size, device=device
        ).to(device)
        try:
            circuit.load_best_params(args.checkpoint)
        except FileNotFoundError:
            circuit.load_current_params(args.checkpoint)
    circuit.eval()

    dim = 2 ** args.num_qubits
    side = int(math.isqrt(dim))
    if side * side != dim:
        raise ValueError(f"Cannot reshape state of dimension {dim} into a square image.")

    os.makedirs(args.output_dir, exist_ok=True)

    final_outputs = np.zeros((args.num_samples, side, side))
    histories = np.zeros((args.num_samples, args.T, side, side))

    for sample_idx in tqdm(range(args.num_samples), desc="Sampling"):
        with torch.no_grad():
            batch = torch.view_as_complex(torch.randn(1, dim, 2, device=device)).to(torch.complex64)
            batch_history = torch.view_as_complex(torch.zeros(args.T, 1, dim, 2, device=device)).to(torch.complex64)

            for t in range(args.T - 1, -1, -1):
                large_batch = torch.zeros(args.T, 1, dim, device=device, dtype=torch.complex64)
                large_batch[t] = batch
                large_batch = large_batch / torch.norm(large_batch.abs(), p=2, dim=2, keepdim=True)
                large_batch = circuit(large_batch)
                batch = large_batch[t, :, :]
                batch_history[t] = batch.unsqueeze(0)
                del large_batch

            final_output = torch.abs(batch).squeeze().cpu().numpy().reshape(side, side)
            final_outputs[sample_idx] = final_output

            batch_history_np = torch.abs(batch_history).squeeze().cpu().numpy().reshape(args.T, side, side)
            histories[sample_idx] = batch_history_np

    grid_side = math.ceil(math.sqrt(args.num_samples))
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(grid_side * 2, grid_side * 2))
    axes_flat = np.array(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i < args.num_samples:
            ax.imshow(final_outputs[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"QuantumDenoising_{args.noise_factor}.png"))
    plt.close(fig)

    fig, axes = plt.subplots(args.num_samples, args.T, figsize=(args.T * 1.5, args.num_samples * 1.5))
    axes_grid = np.array(axes).reshape(args.num_samples, args.T)
    for i in range(args.num_samples):
        for t in range(args.T):
            axes_grid[i, t].imshow(histories[i, t], cmap='gray')
            axes_grid[i, t].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"QuantumDenoisingHistory_{args.noise_factor}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
