# src.utils.loss.py

import torch

class LossHistory:
    def __init__(self, save_dir, data_loader_len):
        self.history = []
        self.save_dir = save_dir
        self.batch_per_epoch = data_loader_len
        self.global_step = 0  # Track the overall number of batches processed

    def log_losses(self, losses, writer):
        # log individual losses
        for i, loss in enumerate(losses):
            writer.add_scalar(f'Loss/loss_{i}', loss.item(), self.global_step)
    
        # compute and log total loss
        tot_loss = losses.mean().item()
        writer.add_scalar('Loss/total_loss', tot_loss, self.global_step)

        # increment global step after logging
        self.global_step += 1

def infidelity_loss(predictions, targets):

    # ensure input and target are complex tensors
    assert torch.is_complex(predictions) and torch.is_complex(targets), "Inputs must be complex tensors."
    
    # shape: (T, BS, 2)
    predictions_norm = predictions / torch.linalg.norm(predictions.abs(), dim=-1, keepdim=True)

    # shape: (T, BS, 2)
    targets_norm = targets / torch.linalg.norm(targets.abs(), dim=-1, keepdim=True)

    # shape: (T, BS)
    fidelity = torch.abs(torch.sum(torch.conj(predictions_norm) * targets_norm, dim=-1)) ** 2

    # return the mean infidelity across the batch
    return 1 - torch.mean(fidelity, dim=-1)
