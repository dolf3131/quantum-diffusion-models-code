# src.utils.training_functions.py

import torch

from src.utils.schedule import get_default_device

device = get_default_device()

# formula (4) in the DDPM paper
def assemble_input(x0, t, alphas_bar):

    # shape: (T*BS)
    alphas_bar_batch = alphas_bar[t]

    # shape: (T*BS, 2)
    avgs = torch.sqrt(alphas_bar_batch).unsqueeze(1).repeat(1, x0.shape[-1])*x0

    # shape: (T*BS, 2)
    fluct = torch.sqrt((1-alphas_bar_batch)).unsqueeze(1).repeat(1, x0.shape[-1])*torch.view_as_complex(torch.randn(x0.shape[0], x0.shape[1], 2, device = device))

    # shape: (T*BS, 2)
    noisy_image = avgs+fluct

    return noisy_image

# formula (7) in the DDPM paper
def assemble_mu_tilde(x0, xt, t, alphas_bar, betas):
      
    # shape: (T*BS)
    coeff_left = torch.sqrt(alphas_bar[t-1])*betas[t]/(1-alphas_bar[t])
    
    # shape: (T*BS)
    coeff_right = torch.sqrt(1-betas[t])*(1-alphas_bar[t-1])/(1-alphas_bar[t])

    return coeff_left.unsqueeze(1).repeat(1, x0.shape[-1])*x0 + coeff_right.unsqueeze(1).repeat(1, x0.shape[-1])*xt
