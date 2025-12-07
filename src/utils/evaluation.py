# src.utils.evaluation.py

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.schedule import get_default_device

# Set device with safe fallback
device = get_default_device()

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load the pre-trained classifier weights
classifier = MNISTClassifier().to(device)
classifier.load_state_dict(torch.load('models/mnist_cnn.pth', map_location=device, weights_only=True))
classifier.eval()

def preprocess_images(generated_images):
    # generated_images: Tensor of shape (N, 256)
    N = generated_images.size(0)
    # Reshape to (N, 1, 16, 16)
    images = generated_images.view(N, 1, 16, 16)
    
    # Normalize images to [0, 1] per image
    images_flat = images.view(N, -1)
    min_vals = images_flat.min(dim=1)[0].view(N, 1, 1, 1)
    max_vals = images_flat.max(dim=1)[0].view(N, 1, 1, 1)
    images = (images - min_vals) / (max_vals - min_vals + 1e-8)
    
    # Resize images to 28x28
    images_resized = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
    
    images_resized = images_resized.to(device)
    return images_resized

def evaluate_generated_images(generated_images):
    images_preprocessed = preprocess_images(generated_images)
    with torch.no_grad():

        # classifier trained on images in range [0, 1]
        outputs = classifier(images_preprocessed)
        probabilities = F.softmax(outputs, dim=1)
        
        # Extract probabilities for zeros and ones
        probabilities_01 = probabilities[:, [0, 1]]
        
        # Add epsilon to probabilities to prevent log(0)
        epsilon = 1e-8
        probabilities_01 = torch.clamp(probabilities_01, min=epsilon, max=1.0)
        
        # Compute the marginal distribution p(y) for zeros and ones
        py_01 = probabilities_01.mean(dim=0)
        
        # Compute the KL divergence for each image
        kl_divergence = probabilities_01 * (probabilities_01.log() - py_01.log())
        kl_divergence = kl_divergence.sum(dim=1)
        
        # Compute the mean KL divergence
        mean_kl_divergence = kl_divergence.mean()
        
        # Compute the Inception Score
        inception_score = torch.exp(mean_kl_divergence)
        
        return inception_score.item()
