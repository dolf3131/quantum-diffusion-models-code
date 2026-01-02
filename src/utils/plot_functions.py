# src.utils.plot_functions.py

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.ansatzes import QuantumUNet
from src.utils.training_functions import assemble_input
from src.utils.schedule import get_default_device

device = get_default_device()


def show_mnist_alphas(mnist_images, alphas_bar, writer, device, height=16, width=16):
    """
    Visualize forward diffusion on a random MNIST sample.
    (This function remains mostly unchanged as it depends on the schedule, not the model architecture)
    """
    mnist_images = mnist_images.to(device)
    # Randomly select one image
    idx = torch.randint(0, mnist_images.shape[0], size=(1,), device=device)
    image = mnist_images[idx].squeeze(0)  # Shape: (D,)

    D = image.numel()
    if height * width != D:
        raise ValueError(f"Cannot reshape image of size {D} into ({height}, {width}).")

    images = [image]
    T = len(alphas_bar)
    
    # Calculate forward diffusion for each step t
    for t in range(T):
        # assemble_input adds noise according to the schedule
        assembled = assemble_input(image.unsqueeze(0), [t], alphas_bar)
        images.append(torch.abs(assembled.squeeze(0)))

    # Plotting
    fig, axs = plt.subplots(2, T + 1, figsize=(20, 6))
    plt.suptitle('Forward diffusion process', fontsize=16)
    bins = np.arange(-0.5, 2.5, 0.2)

    for i in range(T + 1):
        # Image Row
        axs[0, i].imshow(images[i].view(height, width).cpu().detach().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title(f't={i}')

        # Histogram Row
        pixel_values = images[i].cpu().detach().numpy().flatten()
        histogram, _ = np.histogram(pixel_values, bins=bins)
        axs[1, i].bar(bins[:-1], histogram, width=0.2, color='#0504aa', alpha=0.7)
        axs[1, i].set_xlim([-1, 3])
        axs[1, i].set_xlabel('Intensity')
        axs[1, i].set_ylabel('Freq')

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
    model_type="unet", # Default to unet
    num_layers=None,   # Unused in Qiskit UNet but kept for compatibility
    init_variance,
    betas,
    pqc_layers=None,
    activation=False,
    bottleneck_qubits=4, # bottlenect
    use_pooling=False,
    num_samples=16,
    # Below are legacy arguments, kept for signature compatibility
    MLP_depth=None, MLP_width=None, PQC_depth=None, ACT_depth=None, num_ancilla=None, batch_size=64
):
    """
    Run reverse diffusion from noise and log a grid of samples to TensorBoard.
    Updated for Qiskit-based QuantumUNet.
    """
    dim = 2 ** num_qubits
    side = int(math.isqrt(dim))
    if side * side != dim:
        # Cannot reshape to an image grid (e.g., if num_qubits is odd)
        return

    # Use default layers if not provided
    if pqc_layers is None:
        pqc_layers = [4]

    try:
        # Initialize the Qiskit-based QuantumUNet
        # Note: We create a fresh instance to load the weights into
        circuit_clone = QuantumUNet(
            num_qubits=num_qubits,
            layers=pqc_layers,
            T=T,
            init_variance=init_variance,
            betas=betas,
            activation=activation,
            device=device,
            bottleneck_qubits=bottleneck_qubits, # Must be passed
            use_pooling=use_pooling
        ).to(device)

        # Load parameters for the specific epoch
        # The new load_current_params expects 'epoch' to find the correct file
        circuit_clone.load_current_params(directory, epoch=epoch)
        circuit_clone.eval()
        
    except FileNotFoundError:
        print(f"No checkpoint found for epoch {epoch}, skipping logging.")
        return
    except Exception as e:
        print(f"Error initializing model for logging: {e}")
        return

    # Generate Samples
    with torch.no_grad():
        # 1. Start with pure noise (Complex Gaussian)
        batch = torch.randn(num_samples, dim, 2, device=device)
        batch = torch.view_as_complex(batch).to(torch.complex64)
        
        # Normalize initial noise
        batch = batch / (torch.norm(batch, p=2, dim=1, keepdim=True) + 1e-8)

        # 2. Reverse Diffusion Loop
        for t in range(T - 1, -1, -1):
            # Construct input tensor of shape (T, Batch, Dim)
            # Only index 't' is filled with the current batch data
            larger_batch = torch.zeros(T, num_samples, dim, device=device, dtype=torch.complex64)
            larger_batch[t] = batch
            
            # Forward pass (Denoising)
            larger_batch = circuit_clone(larger_batch)
            
            # Update batch with the denoised output at time t
            batch = larger_batch[t]
            
            # Renormalize (Important for quantum states)
            batch = batch / (torch.norm(batch, p=2, dim=1, keepdim=True) + 1e-8)
            
            del larger_batch

        # 3. Process Final Images (Magnitude)
        images = torch.abs(batch).cpu().numpy().reshape(num_samples, side, side)

    # Plotting Grid
    grid_side = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(2 * grid_side, 2 * grid_side))
    
    # Flatten axes for easy iteration
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flat
    else:
        axes_flat = [axes]

    for idx, ax in enumerate(axes_flat):
        if idx < num_samples:
            ax.imshow(images[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    
    # Add to TensorBoard
    writer.add_figure('Generated samples', fig, global_step=epoch)
    plt.close(fig)