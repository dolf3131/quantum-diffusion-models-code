# ansatzes.py

import torch
from torch import nn

from src.models.custom_classes import IdentityUnitaryR3Ansatz, IdentityUnitaryR3Ansatz_batch


def batch_angle_encoding(theta):
    """
    Angle Encoding: 각도(theta)를 양자 상태 벡터로 변환
    theta: (Batch_Size, num_qubits)
    Returns: (Batch_Size, 2^num_qubits)
    """
    batch_size, num_qubits = theta.shape
    device = theta.device
    
    # 초기 상태 |00...0>
    state = torch.ones(batch_size, 1, device=device, dtype=torch.complex64)

    for i in range(num_qubits):
        # Ry(theta) 회전: cos(theta/2)|0> + sin(theta/2)|1>
        angle = theta[:, i].unsqueeze(1) # (Batch, 1)
        
        cos_val = torch.cos(angle / 2)
        sin_val = torch.sin(angle / 2)
        
        # 큐비트 상태: [cos, sin]
        qubit_state = torch.stack([cos_val, sin_val], dim=1).view(batch_size, 2).to(torch.complex64)

        # Tensor Product (state ⊗ qubit_state)
        state = state.view(batch_size, -1, 1) * qubit_state.view(batch_size, 1, 2)
        state = state.view(batch_size, -1)
        
    return state

class ComplexLeakyReLU(nn.Module):
    """LeakyReLU that acts independently on real and imaginary parts."""

    def __init__(self, negative_slope=0.01):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.real_leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)
        self.imag_leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, input):
        return torch.complex(
            self.real_leaky_relu(input.real),
            self.imag_leaky_relu(input.imag)
        )
class ComplexLinear(nn.Module):
    """
    Complex-valued Linear Layer with optional activation.
    """
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.fc = nn.Linear(in_features * 2, out_features)
        self.activation = activation
        self.act_fn = nn.ReLU()

    def forward(self, x_complex):
        x_cat = torch.cat([x_complex.real, x_complex.imag], dim=-1) # (Batch, 2*Dim)
        out = self.fc(x_cat)
        if self.activation:
            out = self.act_fn(out)
        return out


class PQC(nn.Module):
    def __init__(self, num_qubits, layers, T, init_variance, betas, activation=False, device='cuda'):
        super(PQC, self).__init__()

        device = self._resolve_device(device)
        self.T = T
        self.betas = betas
        self.num_qubits = num_qubits
        self.best_metric = [float('-inf')] * T

        if len(layers) != 3:
            raise ValueError("pqc_layers must contain exactly three depths, e.g., [4, 4, 4].")

        layers1, layers2, layers3 = layers

        self.model1 = IdentityUnitaryR3Ansatz(num_qubits, layers1, device=device).to(device)
        self.model2 = IdentityUnitaryR3Ansatz(num_qubits + 1, layers2, device=device).to(device)
        self.model3 = IdentityUnitaryR3Ansatz(num_qubits, layers3, device=device).to(device)

        self.params1 = nn.Parameter(
            torch.randn(T, layers1, num_qubits, 3, requires_grad=True, device=device) * init_variance
        )
        self.params2 = nn.Parameter(
            torch.randn(T, layers2, num_qubits + 1, 3, requires_grad=True, device=device) * init_variance
        )
        self.params3 = nn.Parameter(
            torch.randn(T, layers3, num_qubits, 3, requires_grad=True, device=device) * init_variance
        )

        self.activation = activation
        if activation:
            self.act_func = ComplexLeakyReLU()

    @staticmethod
    def _resolve_device(device):
        device = torch.device(device)
        if device.type == 'cuda':
            try:
                if not torch.cuda.is_available() or torch.version.cuda is None:
                    return torch.device('cpu')
            except Exception:
                return torch.device('cpu')
        return device

    def _save_param_set(self, directory, filename, params1, params2, params3):
        torch.save(
            {
                'params1': params1.detach().clone(),
                'params2': params2.detach().clone(),
                'params3': params3.detach().clone()
            },
            f'{directory}/{filename}'
        )

    def save_params(self, directory, epoch=None, best=False):
        """
        Save current parameters.
        - If best=True, save to best{i}.pt.
        - If epoch is provided, append the epoch to the filename.
        - Otherwise, mirror NNCPQC naming with current{i}.pt.
        """
        for i in range(self.T):
            filename = f'current{i}.pt'
            if best:
                filename = f'best{i}.pt'
            elif epoch is not None:
                filename = f'epoch{epoch}_T{i}.pt'
            self._save_param_set(directory, filename, self.params1[i], self.params2[i], self.params3[i])

    def save_current_params(self, directory, epoch=None):
        """Alias for backward compatibility."""
        self.save_params(directory, epoch=epoch, best=False)

    def _load_params(self, path, T_index, noise=None):
        loaded_params = torch.load(path)

        updated_params1 = self.params1.clone().detach()
        updated_params2 = self.params2.clone().detach()
        updated_params3 = self.params3.clone().detach()

        noise_term1 = noise * torch.randn_like(loaded_params['params1']) if noise is not None else 0.0
        noise_term2 = noise * torch.randn_like(loaded_params['params2']) if noise is not None else 0.0
        noise_term3 = noise * torch.randn_like(loaded_params['params3']) if noise is not None else 0.0

        updated_params1[T_index] = loaded_params['params1'].to(self.params1.device) + noise_term1
        updated_params2[T_index] = loaded_params['params2'].to(self.params2.device) + noise_term2
        updated_params3[T_index] = loaded_params['params3'].to(self.params3.device) + noise_term3

        self.params1 = nn.Parameter(updated_params1, requires_grad=True)
        self.params2 = nn.Parameter(updated_params2, requires_grad=True)
        self.params3 = nn.Parameter(updated_params3, requires_grad=True)

    def load_params_from_epoch(self, directory, epoch, T_index, noise=None):
        path = f'{directory}/epoch{epoch}_T{T_index}.pt'
        self._load_params(path, T_index=T_index, noise=noise)

    def load_current_params(self, directory, epoch=None, noise=None):
        """
        Load parameters saved via save_params/save_current_params.
        Supports both current{i}.pt naming (epoch=None) and epoch{epoch}_T{i}.pt.
        """
        for i in range(self.T):
            if epoch is None:
                path = f'{directory}/current{i}.pt'
            else:
                path = f'{directory}/epoch{epoch}_T{i}.pt'
            self._load_params(path, T_index=i, noise=noise)

    def load_best_params(self, directory, noise=None):
        for i in range(self.T):
            adjusted_noise = self.betas[i] * noise if noise is not None else None
            path = f'{directory}/best{i}.pt'
            self._load_params(path, T_index=i, noise=adjusted_noise)

    def update_best_params(self, directory, metric):
        """
        Save current parameters as best if the provided metric improves.
        The metric can be any scalar (e.g., inception score or -loss).
        """
        for i in range(self.T):
            if metric > self.best_metric[i]:
                self.best_metric[i] = metric
                self._save_param_set(directory, f'best{i}.pt', self.params1[i], self.params2[i], self.params3[i])

    def forward(self, input):
        batch_size = input.shape[1]

        # reshape input to have a single batch dimension
        input = input.reshape(self.T * batch_size, -1)

        # first PQC block on data qubits
        output1 = self.model1(input, self.params1)

        # append ancilla |0> for middle block
        output_with_ancilla = torch.cat((output1, torch.zeros_like(output1)), dim=1)

        # second PQC block on data+ancilla
        output2 = self.model2(output_with_ancilla, self.params2)

        # measure ancilla qubit and accept remaining qubits only if ancilla is |0>
        output2 = output2[:, :2 ** self.num_qubits]

        # renormalize after measurement
        output2 = output2 / torch.norm(output2.abs(), p=2, dim=1, keepdim=True)

        # optional activation
        if self.activation:
            output2 = self.act_func(output2)
            output2 = output2 / torch.norm(output2.abs(), p=2, dim=1, keepdim=True)

        # third PQC block on data qubits
        predictions = self.model3(output2, self.params3)
        predictions = predictions / torch.norm(predictions.abs(), p=2, dim=1, keepdim=True)
        # truncate just in case (Identity ansatz preserves width, but keep consistency)
        predictions = predictions[:, :2 ** self.num_qubits]

        # reshape the predictions to match the batch size
        out = predictions.reshape(self.T, batch_size, 2 ** self.num_qubits)

        return out

    def get_pqc_params(self):
        return [self.params1, self.params2, self.params3]

    def get_mlp_params(self):
        # For interface compatibility with NNCPQC
        return []


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        
        # Create a list to store layers
        layers = []
        
        # Add hidden layers with ReLU activations
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim  # Update input_dim for the next layer
        
        # Add output layer with Sigmoid activation
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        # Combine all layers into a Sequential module
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class NNCPQC(nn.Module):
    def __init__(self, num_qubits, num_ancilla, num_layers, MLP_depth, MLP_width, PQC_depth, ACT_depth, T, init_variance, batch_size, device='cuda'):
        super(NNCPQC, self).__init__()

        device = PQC._resolve_device(device)
        self.T = T
        PQC_width = num_qubits + num_ancilla
        self.ACT_width = num_qubits + num_ancilla - 1
        self.ACT_depth = ACT_depth

        self.PQC_models = nn.ModuleList()
        self.ACT_models = nn.ModuleList()
        self.MLP_models = nn.ModuleList()

        self.PQC_params = nn.ParameterList()

        self.best_losses = [float('inf')] * T
        self.num_ancilla = num_ancilla
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        input_dim_MLP = 2 ** (num_qubits)
        output_dim = self.ACT_width * ACT_depth * num_layers * 3


        # MLP layers
        for _ in range(T):
            self.MLP_models.append(nn.Sequential(MLP(input_dim_MLP, [MLP_width] * MLP_depth, output_dim).to(device)))

        # PQC layers
        for _ in range(num_layers):
            self.PQC_models.append(IdentityUnitaryR3Ansatz(PQC_width, PQC_depth, device=device).to(device))
            self.PQC_params.append(nn.Parameter(torch.randn(T, PQC_depth, PQC_width, 3, requires_grad=True, device=device) * init_variance))
        
        # ACT layers
        for _ in range(num_layers):
            self.ACT_models.append(IdentityUnitaryR3Ansatz_batch(self.ACT_width, ACT_depth, batch_size, device=device).to(device))

    def forward(self, input):

        # input shape: (T, BS, 2^num_qubits)

        T, batch_size = input.shape[0], input.shape[1]

        # generate activation parameters
        all_ACT_params = []
        for t in range(self.T):
            input_MLP = input[t]
            input_MLP = torch.abs(input_MLP)

            # shape: (64, ACT_width * ACT_depth * num_layers * 3)
            rotation_params = self.MLP_models[t](input_MLP)*2*torch.pi

            all_ACT_params.append(rotation_params)

        # shape: (T, 64, num_layers * ACT_depth * ACT_width * 3)
        all_ACT_params = torch.stack(all_ACT_params, dim=0)

        # shape: (T, BS, num_layers, ACT_depth, ACT_width, 3)
        all_ACT_params = all_ACT_params.reshape(T, batch_size, self.num_layers, self.ACT_depth, self.ACT_width, 3)

        # increase input with ancillas
        # shape: (T*BS, 2^num_qubits)
        input = input.reshape(T * batch_size, -1)

        for _ in range(self.num_ancilla): 
            input = torch.cat((input, torch.zeros_like(input)), dim=1)
        
        # shape: (T*BS, 2^PQC_width)
        output = input

        # feed through layers
        for layer in range(self.num_layers):

            # slice parameters for activation layers
            # shape: (T, BS, ACT_depth, ACT_width, 3)
            ACT_params = all_ACT_params[:,:,layer,:,:,:]

            # shape: (T, PQC_depth, PQC_width, 3)
            PQC_params = self.PQC_params[layer]

            # PQC block
            # shape: (T*BS, 2^PQC_width)
            output = self.PQC_models[layer](output, PQC_params)

            # measure ancilla and normalize
            # shape: (T*BS, 2^num_qubits)
            output = output[:, :2 ** self.ACT_width]
            output = output / torch.norm(output.abs(), p=2, dim=1, keepdim=True)

            # activation block
            # shape: (T*BS, 2^num_qubits)
            output = self.ACT_models[layer](output, ACT_params)
            output = output / torch.norm(output.abs(), p=2, dim=1, keepdim=True)

            # put back ancilla
            # shape: (T*BS, 2^PQC_width)
            output = torch.cat((output, torch.zeros_like(output)), dim=1)

        # undo last cat (trick to write everything in the for loop)
        # shape: (T*BS, 2^num_qubits)
        output = output[:, :2 ** self.ACT_width]

        # final reshape
        # shape: (T, 64, 2^num_qubits)
        out = output.reshape(T, batch_size, 2 ** self.num_qubits)

        return out

    def save_params(self, directory, best=False):
        prefix = 'best' if best else 'current'
        for i in range(self.T):
            params = {f'pqc_params{j+1}': PQC_param[i].detach().clone() for j, PQC_param in enumerate(self.PQC_params)}
            mlp_params = {f'mlp_params{j+1}': MLP_model.state_dict() for j, MLP_model in enumerate(self.MLP_models)}
            torch.save({**params, **mlp_params}, f'{directory}/{prefix}{i}.pt')

    def update_best_params(self, directory, losses):
        for i in range(self.T):
            current_loss = losses[i]
            if current_loss < self.best_losses[i]:
                self.best_losses[i] = current_loss
                params = {f'pqc_params{j+1}': PQC_param[i].detach().clone() for j, PQC_param in enumerate(self.PQC_params)}
                mlp_params = {f'mlp_params{j+1}': MLP_model.state_dict() for j, MLP_model in enumerate(self.MLP_models)}
                torch.save({**params, **mlp_params}, f'{directory}/best{i}.pt')

    def load_params(self, path, copy_index):
        loaded_params = torch.load(path, weights_only=True)
        for j, model in enumerate(self.MLP_models):
            model.load_state_dict(loaded_params[f'mlp_params{j+1}'])  

        updated_params = []
        for j, param in enumerate(self.PQC_params):
            cloned_param = param.clone().detach()
            assert cloned_param[copy_index].shape == loaded_params[f'pqc_params{j+1}'].shape, \
                "Shape mismatch during parameter loading."
            cloned_param[copy_index] = loaded_params[f'pqc_params{j+1}'].to(param.device)
            updated_params.append(nn.Parameter(cloned_param, requires_grad=True))
        self.PQC_params = nn.ParameterList(updated_params)

    def load_best_params(self, directory):
        for i in range(self.T):
            path = f'{directory}/best{i}.pt'
            self.load_params(path, copy_index=i)

    def load_current_params(self, directory):
        for i in range(self.T):
            path = f'{directory}/current{i}.pt'
            self.load_params(path, copy_index=i)

    def get_pqc_params(self):
        # Returns the PQC parameters for separate optimization
        return list(self.PQC_params)

    def get_mlp_params(self):
        # Returns the MLP parameters for separate optimization
        mlp_parameters = []
        for model in self.MLP_models: mlp_parameters.extend(list(model.parameters()))
        return mlp_parameters


class QuantumUNet(nn.Module):
    """
    Hybrid Quantum-Classical U-Net Ansatz
    1. Encoder: ComplexLinear layers to downsample input to bottleneck size
    2. Bottleneck: PQC operating on reduced qubit count
    3. Decoder: ComplexLinear layers with skip connections to reconstruct output
    4. Final output layer projecting back to original dimension
    5. Normalization at key steps to maintain valid quantum states
    6. Skip connections from encoder to decoder for feature preservation
    7. PQC parameters managed via inherited methods from PQC class
    8. Designed for quantum diffusion models with reduced quantum resource requirements
    9. Flexible bottleneck qubit count for trade-off between expressivity and resource use
    10. Suitable for image-like quantum data (e.g., 8 qubits = 256 dim input)
    """
    def __init__(self, num_qubits, layers, T, init_variance, betas, activation=False, device='cuda', bottleneck_qubits=4):
        super(QuantumUNet, self).__init__()
        
        device = PQC._resolve_device(device)
        self.device = device
        self.T = T
        self.betas = betas
        self.num_qubits = num_qubits  # 원본 이미지 큐비트 (예: 8 -> 256 dim)
        self.bn_qubits = bottleneck_qubits # 병목 구간 큐비트 (예: 4 -> 16 dim)
        
        input_dim = 2 ** num_qubits
        input_flat_dim = input_dim * 2   # 512 (Real + Imag)
        hidden_dim = 64 
        bn_dim = 2 ** bottleneck_qubits

        # --- [1] 활용할 PQC 모듈 초기화 (Composition) ---
        # 작은 큐비트 수(bottleneck_qubits)를 가진 PQC를 생성하여 내부에 둡니다.
        # 이렇게 하면 기존 PQC의 검증된 forward 로직을 그대로 쓸 수 있습니다.
        self.pqc_bottleneck = PQC(
            num_qubits=bottleneck_qubits, 
            layers=layers, 
            T=T, 
            init_variance=init_variance, 
            betas=betas, 
            activation=activation, 
            device=device
        )

        # --- [2] Encoder (Downsampling) ---
        # ComplexLinear를 사용하여 Phase 정보를 잃지 않도록 함
        self.enc1 = ComplexLinear(input_dim, hidden_dim) 
        self.enc2 = nn.Linear(hidden_dim, bn_dim * 2)

        # --- [3] Decoder (Upsampling with Skip) ---
        # self.dec1 = nn.Linear(bn_dim * 2 + hidden_dim, hidden_dim) 
        # self.act_dec1 = nn.ReLU()
        self.dec1 = nn.Linear(bn_dim * 2 + input_flat_dim, hidden_dim) 
        self.act_dec1 = nn.ReLU()
        # Final Output: 256 dim (Complex 출력을 위해 2배인 512 출력)
        # 지름길 학습 방지를 위해 Input Skip은 제거
        self.final = nn.Linear(hidden_dim, input_dim * 2)

    def forward(self, input):
        # input: (T, Batch, 256) Complex
        T_steps, batch_size, dim = input.shape
        
        # 1. Flatten Input (Real + Imag)
        x = input.reshape(T_steps * batch_size, dim)
        # 원본 입력을 실수 벡터로 펼칩니다. (나중에 Skip으로 씀)
        x_flat = torch.cat([x.real, x.imag], dim=-1) # (Batch, 512)

        # --- Encoder ---
        e1 = self.enc1(x)        # (Batch, 64)
        latent = self.enc2(e1)   # (Batch, 2*bn_dim)
        
        # --- Quantum State Prep ---
        bn_dim = 2 ** self.bn_qubits
        state = torch.complex(latent[:, :bn_dim], latent[:, bn_dim:])
        state = state / (torch.norm(state, p=2, dim=1, keepdim=True) + 1e-8)
        
        # --- PQC Execution ---
        state_reshaped = state.reshape(T_steps, batch_size, -1)
        pqc_out = self.pqc_bottleneck(state_reshaped)
        pqc_out = pqc_out.reshape(T_steps * batch_size, -1)

        # --- Decoder ---
        pqc_feats = torch.cat([pqc_out.real, pqc_out.imag], dim=-1)
        
        # [핵심 변경] e1 대신 x_flat(원본)을 결합합니다.
        # 이제 Decoder는 '압축된 정보'와 '원본 정보'를 동시에 봅니다.
        d1_input = torch.cat([pqc_feats, x_flat], dim=1) 
        
        d1 = self.act_dec1(self.dec1(d1_input))
        out_feats = self.final(d1)
        
        # Output Reconstruction
        out_real = out_feats[:, :dim]
        out_imag = out_feats[:, dim:]
        out = torch.complex(out_real, out_imag)
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-8)
        
        return out.reshape(T_steps, batch_size, dim)
    
    # --- PQC의 파라미터 관리 기능 연동 (상속 효과) ---
    def get_pqc_params(self):
        return self.pqc_bottleneck.get_pqc_params() + \
               list(self.enc1.parameters()) + list(self.enc2.parameters()) + \
               list(self.dec1.parameters()) + list(self.final.parameters())

    def save_params(self, directory, epoch=None, best=False):
        for i in range(self.T):
            filename = f'best{i}.pt' if best else (f'current{i}.pt' if epoch is None else f'epoch{epoch}_T{i}.pt')
            pqc_params = {
                'params1': self.pqc_bottleneck.params1[i].detach().clone(),
                'params2': self.pqc_bottleneck.params2[i].detach().clone(),
                'params3': self.pqc_bottleneck.params3[i].detach().clone(),
                'unet_state': self.state_dict()
            }
            torch.save(pqc_params, f'{directory}/{filename}')
            
    def load_current_params(self, directory, epoch=None, noise=None):
        i = self.T - 1
        path = f'{directory}/current{i}.pt' if epoch is None else f'{directory}/epoch{epoch}_T{i}.pt'
        try:
            loaded_params = torch.load(path, map_location=self.device)
            if 'unet_state' in loaded_params:
                self.load_state_dict(loaded_params['unet_state'])
        except FileNotFoundError:
            pass

    def get_mlp_params(self): return []
    
