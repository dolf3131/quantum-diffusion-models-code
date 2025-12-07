# src.utils.plot_functions.py

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.ansatzes import NNCPQC, PQC
from src.utils.training_functions import assemble_input
from src.utils.schedule import get_default_device

device = get_default_device()


def show_mnist_alphas(mnist_images, alphas_bar, writer, device, height=16, width=16):
    """Visualize forward diffusion on a random MNIST sample."""
    mnist_images = mnist_images.to(device)
    idx = torch.randint(0, mnist_images.shape[0], size=(1,), device=device)
    image = mnist_images[idx].squeeze(0)  # Shape: (D,)

    D = image.numel()
    if height * width != D:
        raise ValueError(f"Cannot reshape image of size {D} into ({height}, {width}).")

    images = [image]
    T = len(alphas_bar)
    for t in range(T):
        assembled = assemble_input(image.unsqueeze(0), [t], alphas_bar)
        images.append(torch.abs(assembled.squeeze(0)))

    fig, axs = plt.subplots(2, T + 1, figsize=(20, 6))
    plt.suptitle('Forward diffusion with assemble_input()', fontsize=16)
    bins = np.arange(-0.5, 2.5, 0.2)

    for i in range(T + 1):
        axs[0, i].imshow(images[i].view(height, width).cpu().detach().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title(f't={i}')

        pixel_values = images[i].cpu().detach().numpy().flatten()
        histogram, _ = np.histogram(pixel_values, bins=bins)
        axs[1, i].bar(bins[:-1], histogram, width=0.2, color='#0504aa', alpha=0.7)
        axs[1, i].set_xlim([-1, 3])
        axs[1, i].set_xlabel('Pixel Intensity')
        axs[1, i].set_ylabel('Frequency')

    for ax in axs.flatten():
        ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    writer.add_figure('Forward diffusion (MNIST)', fig)
    plt.close(fig)


def log_generated_samples(
    directory,
    epoch,
    T,
    num_qubits,
    writer,
    *,
    model_type,
    num_layers,
    MLP_depth,
    MLP_width,
    PQC_depth,
    ACT_depth,
    num_ancilla,
    init_variance,
    betas,
    pqc_layers=None,
    activation=False,
    batch_size=64,
    num_samples=16,
):
    """Run reverse diffusion from noise and log a grid of samples to TensorBoard."""
    dim = 2 ** num_qubits
    side = int(math.isqrt(dim))
    if side * side != dim:
        # Cannot reshape to an image grid.
        return

    model_type = model_type.lower()
    try:
        if model_type == 'pqc':
            layers = pqc_layers if pqc_layers is not None else [PQC_depth, PQC_depth, PQC_depth]
            circuit_clone = PQC(
                num_qubits, layers, T, init_variance, betas, activation=activation, device=device
            ).to(device)
            circuit_clone.load_current_params(directory)
        else:
            circuit_clone = NNCPQC(
                num_qubits, num_ancilla, num_layers, MLP_depth, MLP_width,
                PQC_depth, ACT_depth, T, init_variance, batch_size, device=device
            ).to(device)
            circuit_clone.load_current_params(directory)
        circuit_clone.eval()
    except FileNotFoundError:
        # No checkpoint yet; skip logging.
        return

    with torch.no_grad():
        batch = torch.view_as_complex(torch.randn(num_samples, dim, 2, device=device)).to(torch.complex64)
        for t in range(T - 1, -1, -1):
            larger_batch = torch.zeros(T, num_samples, dim, device=device, dtype=torch.complex64)
            larger_batch[t] = batch
            larger_batch = larger_batch / torch.norm(larger_batch, p=2, dim=2, keepdim=True)
            larger_batch = circuit_clone(larger_batch)
            batch = larger_batch[t]
            del larger_batch

        images = torch.abs(batch).cpu().numpy().reshape(num_samples, side, side)

    grid_side = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(2 * grid_side, 2 * grid_side))
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            ax.imshow(images[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    writer.add_figure('Generated samples', fig, global_step=epoch)
    plt.close(fig)
