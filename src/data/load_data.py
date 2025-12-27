import os
import torch
import torch.nn.functional as F
from torchvision import datasets

def load_mnist(desired_digits, data_length=None, device=None):
    """
    Processes the MNIST dataset by selecting desired digits, normalizing, resizing,
    shuffling, and saving the processed data as a tensor.

    Args:
        desired_digits (list of int): List of digits to include (e.g., [0, 1, 2]).
        data_length (int, optional): Number of samples to include. If None, include all.
        device (str or torch.device, optional): Preferred device. Defaults to 'cuda' when available, otherwise 'cpu'.

    Returns:
        torch.Tensor: The processed dataset tensor.
    """
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Determine the directory one level up from the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '../..', 'data'))

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Check if MNIST is already downloaded; if not, download it to data_dir
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True)

    data = mnist_dataset.data.float()  # Shape: (60000, 28, 28)
    targets = mnist_dataset.targets  # Shape: (60000,)

    # Select desired digits
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for digit in desired_digits:
        mask |= (targets == digit)
    selected_data = data[mask]  # Shape: (N_selected, 28, 28)

    if selected_data.size(0) == 0:
        raise ValueError("No data found for the specified digits.")

    # Convert to float and normalize each image to have L2 norm = 1
    selected_data = selected_data.view(selected_data.size(0), -1)  # Flatten to (N_selected, 784)
    selected_data = F.normalize(selected_data, p=2, dim=1)  # Normalize to L2 norm = 1

    # Resize images to 16x16
    selected_data = selected_data.view(-1, 1, 28, 28)  # Reshape to (N_selected, 1, 28, 28)
    selected_data = F.interpolate(selected_data, size=(16, 16), mode='bilinear', align_corners=False)
    selected_data = selected_data.view(selected_data.size(0), -1)  # Flatten to (N_selected, 256)

    # Shuffle the data
    perm = torch.randperm(selected_data.size(0))
    selected_data = selected_data[perm]

    # If data_length is specified, take the first data_length samples
    if data_length is not None:
        if data_length > selected_data.size(0):
            raise ValueError(f"Requested data_length {data_length} exceeds available samples {selected_data.size(0)}.")
        selected_data = selected_data[:data_length]

    # Move to the chosen device for downstream training
    selected_data = selected_data.to(device)

    print(f"Full MNIST dataset is stored in: {data_dir}")

    # Save the processed data, overwriting any existing file
    save_path = os.path.join(data_dir, 'mnist_dataset.pth')
    torch.save(selected_data.cpu(), save_path)
    print(f"Dataset is stored in: {save_path}")

    # Return the path to the saved dataset
    return selected_data
