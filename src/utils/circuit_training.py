# src.utils.circuit_training.py

import os
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.ansatzes import NNCPQC, PQC, QuantumUNet
from src.utils.schedule import make_schedule, device
from src.utils.loss import infidelity_loss, LossHistory
from src.utils.training_functions import assemble_input, assemble_mu_tilde
from src.utils.plot_functions import show_mnist_alphas, log_generated_samples
from src.data.load_data import load_mnist

def training(path, hyperparameters, data_length):

    # release memory
    torch.cuda.empty_cache()
    
    # Unpack hyperparameters
    (
        model_type, num_layers, PQC_LR, MLP_LR, batch_size, num_epochs, scheduler_patience, scheduler_gamma,
        T, num_qubits, beta0, betaT, schedule, schedule_exponent, init_variance, wd_PQC, wd_MLP,
        desired_digits, inference_noise, load_epoch, activation,
        MLP_width, MLP_depth, PQC_depth, ACT_depth, num_ancilla, checkpoint, pqc_layers, bottleneck_qubits
    ) = hyperparameters

    model_type = model_type.lower()
    is_pqc = model_type == 'pqc'

    betas, alphas_bar = make_schedule(beta0, betaT, T, schedule, schedule_exponent, device)
    best_loss = float('inf')
    sample_log_interval = 50

    tensorboard_dir = os.path.join(path, 'TensorBoard')
    params_dir = os.path.join(path, 'Params')
    logs_dir = os.path.join(path, 'Logs')

    # Log all hyperparameters to tensorboard
    writer = SummaryWriter(tensorboard_dir)
    writer.add_hparams(
        {
            'model_type': model_type, 'num_layers': num_layers, 'PQC_LR': PQC_LR, 'MLP_LR': MLP_LR, 'batch_size': batch_size,
            'num_epochs': num_epochs, 'scheduler_patience': scheduler_patience,
            'scheduler_gamma': scheduler_gamma, 'time steps': T, 'num_qubits': num_qubits,
            'beta0': beta0, 'betaT': betaT, 'schedule': schedule,
            'schedule_exponent': schedule_exponent, 'wd_PQC': wd_PQC, 'wd_MLP': wd_MLP,
            'init_variance': init_variance, 'desired_digits': torch.tensor(desired_digits),
            'inference_noise': inference_noise, 'load_epoch': load_epoch,
            'activation': activation, 'MLP_width': MLP_width, 'MLP_depth': MLP_depth,
            'PQC_depth': PQC_depth, 'ACT_depth': ACT_depth, 'num_ancilla': num_ancilla, 'checkpoint': checkpoint,
            'pqc_layers': torch.tensor(pqc_layers) if pqc_layers is not None else torch.tensor([])
        },
        {'hparam/best_loss': best_loss}
    )

    # open necessary folders
    os.makedirs(path, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # load dataset, show forward process
    dataset = load_mnist(desired_digits, data_length)
    show_mnist_alphas(dataset, alphas_bar, writer, device, height=16, width=16)

    if is_pqc:
        layers = pqc_layers if pqc_layers is not None else [PQC_depth, PQC_depth, PQC_depth]
        # circuit = PQC(num_qubits, layers, T, init_variance, betas, activation=activation, device=device).to(device)
        # optimizer = Adam([{'params': circuit.get_pqc_params(), 'lr': PQC_LR, 'weight_decay': wd_PQC}])
        layers = pqc_layers if pqc_layers is not None else [PQC_depth, PQC_depth, PQC_depth]
        circuit = QuantumUNet(
            num_qubits=num_qubits,       # 입력: 8 (256차원)
            layers=layers,            # PQC 레이어 깊이
            T=T, 
            init_variance=init_variance, 
            betas=betas, 
            activation=activation, 
            device=device,
            bottleneck_qubits=bottleneck_qubits         # 병목: 4 (16차원) - 조절 가능
        ).to(device)
        optimizer = Adam([{'params': circuit.get_pqc_params(), 'lr': PQC_LR, 'weight_decay': wd_PQC}])
    else:
        circuit = NNCPQC(num_qubits, num_ancilla, num_layers, MLP_depth, MLP_width, PQC_depth, ACT_depth, T, init_variance, batch_size).to(device)
        optimizer = Adam([
            {'params': circuit.get_pqc_params(), 'lr': PQC_LR, 'weight_decay': wd_PQC},
            {'params': circuit.get_mlp_params(), 'lr': MLP_LR, 'weight_decay': wd_MLP}
        ])

    # load previous checkpoint if provided
    if checkpoint is not None:
        param_dir = checkpoint if os.path.isabs(checkpoint) else os.path.join(os.getcwd(), checkpoint)
        if is_pqc:
            if load_epoch == 'best':
                circuit.load_best_params(param_dir, noise=inference_noise)
            elif load_epoch is not None:
                circuit.load_current_params(param_dir, epoch=load_epoch, noise=inference_noise)
            else:
                circuit.load_current_params(param_dir, noise=inference_noise)
        else:
            if load_epoch == 'best':
                circuit.load_best_params(param_dir)
            else:
                circuit.load_current_params(param_dir)

    # make dataloader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # scheduler, loss monitor and loss plotter
    scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_patience, gamma=scheduler_gamma)
    loss_monitor = LossHistory(logs_dir, len(data_loader))
    
    # training loop
    for epoch in range(1, num_epochs + 1):

        lr = scheduler.get_last_lr()
        epoch_losses = []

        epoch_progress_bar = tqdm(
        enumerate(data_loader), 
        total=len(data_loader), 
        desc=f"Epoch {epoch}/{num_epochs}")

        # shape: (BS, 2^num_qubits)
        for image_batch in data_loader:

            optimizer.zero_grad()

            # shape: (BS*T)
            t = torch.tensor(range(1, T+1), dtype=torch.long).repeat_interleave(batch_size).to(device)

            # shape: (BS*T, 2^num_qubits)
            image_batch = image_batch.repeat(T, 1).to(device)
            
            input_batch = assemble_input(image_batch, t, alphas_bar).to(torch.complex64)
            mu_tilde_t  = assemble_mu_tilde(image_batch, input_batch, t, alphas_bar, betas).to(torch.complex64)

            # Reshape
            input_batch = input_batch.view(T, batch_size, -1)
            mu_tilde_t  = mu_tilde_t.view(T, batch_size, -1)

            # Normalize
            eps = 1e-8
            input_batch = input_batch / (torch.norm(input_batch.abs(), p=2, dim=-1, keepdim=True) + eps)
            mu_tilde_t  = mu_tilde_t / (torch.norm(mu_tilde_t.abs(), p=2, dim=-1, keepdim=True) + eps)

            # shape: (T, BS, 2^num_qubits)
            predicted_mu_t = circuit(input_batch)

            # shape: (T,)
            losses = infidelity_loss(predicted_mu_t, mu_tilde_t)

            # save history
            loss_monitor.log_losses(losses, writer)

            # Backpropagation and optimization
            loss = torch.mean(losses)
            loss.backward()
            optimizer.step()

            # Update the progress bar with the current loss
            epoch_progress_bar.set_postfix({'Loss': loss.item()})
            epoch_progress_bar.update(1)
            epoch_losses.append(loss.detach().item())

        # log epoch number and
        writer.add_scalar('Epoch', epoch, epoch)

        # adjust learning rate
        scheduler.step()

        # get LRs
        current_lrs = scheduler.get_last_lr()

        # Log the learning rates explicitly
        writer.add_scalar('Learning Rate/PQC', current_lrs[0], epoch)
        if not is_pqc and len(current_lrs) > 1:
            writer.add_scalar('Learning Rate/MLP', current_lrs[1], epoch)

        # Optional: Print if learning rates have changed
        if current_lrs != lr:
            if not is_pqc and len(current_lrs) > 1:
                print(f'Learning rates changed - PQC: {current_lrs[0]}, MLP: {current_lrs[1]}')
            else:
                print(f'Learning rate changed - PQC: {current_lrs[0]}')

        # save current params
        circuit.save_params(params_dir, best=False)

        if epoch % sample_log_interval == 0:
            log_generated_samples(
                params_dir, epoch, T, num_qubits, writer, model_type=model_type,
                num_layers=num_layers, MLP_depth=MLP_depth, MLP_width=MLP_width,
                PQC_depth=PQC_depth, ACT_depth=ACT_depth, num_ancilla=num_ancilla,
                init_variance=init_variance, betas=betas, pqc_layers=pqc_layers,
                activation=activation, batch_size=batch_size, bottleneck_qubits=bottleneck_qubits
            )

        epoch_loss_value = float(np.mean(epoch_losses))
        writer.add_scalar('Loss/epoch_mean', epoch_loss_value, epoch)

        if epoch_loss_value < best_loss:
            best_loss = epoch_loss_value
            writer.add_scalar('Best Total Loss', best_loss, epoch)
            circuit.save_params(params_dir, best=True)
