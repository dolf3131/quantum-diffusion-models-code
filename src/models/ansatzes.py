import torch
from torch import nn
import numpy as np

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

backend_sim = AerSimulator()
pm = generate_preset_pass_manager(target=backend_sim.target, optimization_level=3, seed_transpiler=42)

# PennyLane
import pennylane as qml

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
    """Complex-valued Linear Layer."""
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.fc = nn.Linear(in_features * 2, out_features) # Output is Real for simplicity in next steps
        self.activation = activation
        self.act_fn = nn.ReLU()

    def forward(self, x_complex):
        # Concatenate Real and Imag parts
        x_cat = torch.cat([x_complex.real, x_complex.imag], dim=-1) 
        out = self.fc(x_cat)
        if self.activation:
            out = self.act_fn(out)
        return out


# --- [2] Qiskit based PQC Layer ---

class QiskitPQCLayer(nn.Module):
    """
    Qiskit 기반의 Variational Quantum Circuit Layer.
    - FeatureMap: 고전 데이터(Latent vector)를 양자 상태로 인코딩 (Angle Encoding)
    - Ansatz: 학습 가능한 파라미터를 가진 회로 (RealAmplitudes)
    - TorchConnector: PyTorch의 autograd와 Qiskit을 연결
    """
    def __init__(self, num_qubits, reps=2):
        super(QiskitPQCLayer, self).__init__()
        self.num_qubits = num_qubits
        
        # 1. Feature Map
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement='linear')
        
        # 2. Ansatz
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement='linear')
        
        # 3. Quantum Circuit
        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        observables = []
        for i in range(num_qubits):
            op_list = ["I"] * num_qubits
            op_list[num_qubits - 1 - i] = "Z"
            op_str = "".join(op_list)
            
            # Pauli Operator 생성
            observables.append(SparsePauliOp(op_str))

        aer_estimator = AerEstimator()

        isa_qc = pm.run(qc)

        # 5. QNN
        self.qnn = EstimatorQNN(
            circuit=isa_qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            observables=observables, 
            input_gradients=True,
            estimator=aer_estimator 
        )
        
        # 6. Torch Connector
        self.qnn_torch = TorchConnector(self.qnn)

    def forward(self, x):
        # x: (Batch, num_qubits) -> QNN -> (Batch, num_qubits)
        # QNN의 출력은 실수(Real)입니다.
        return self.qnn_torch(x)


def ConvUnit(params, wires):
    """
    User-defined 2-Qubit Convolution Layer (3 params)
    Args:
        params: Flat tensor containing all parameters
        wires: [control, target]
        param_idx: Current index pointer in params
    Returns:
        Updated param_idx
    """
    param_idx = 0
    
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    
    qml.RZ(params[param_idx], wires=wires[0]); param_idx += 1
    qml.RY(params[param_idx], wires=wires[1]); param_idx += 1
    
    qml.CNOT(wires=[wires[0], wires[1]])
    
    qml.RY(params[param_idx], wires=wires[1]); param_idx += 1
    
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi/2, wires=wires[0])
    
    return param_idx

def MixingBlock(params, wires):
    """
    RX + CRY based variational ansatz
    Input params: 1D Tensor for this block
    """
    n = len(wires)
    param_idx = 0
    
    # 1. RX Layer (n params)
    for i in range(n):
        qml.RX(params[param_idx], wires=wires[i])
        param_idx += 1
    
    # 2. CRY Layer (Pairwise)
    
    # Even pairs: (0,1), (2,3)...
    for i in range(0, n-1, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        qml.RZ(params[param_idx], wires=wires[(i+1)%n]); param_idx += 1
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        
    # Odd pairs: (1,2), (3,4)...
    for i in range(1, n, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        qml.RZ(params[param_idx], wires=wires[(i+1)%n]); param_idx += 1
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
    
    return param_idx

def ZZInteractionLayer(params, wires):
    """
    [Phase Correlation]
    ZZ gate
    """
    n = len(wires)
    param_idx = 0
    for i in range(0, n - 1, 2):
        w1, w2 = wires[i], wires[i+1]
        # ZZ Gate Decomposition: CNOT -> RZ -> CNOT
        qml.CNOT(wires=[w1, w2])
        qml.RZ(params[param_idx], wires=w2)
        qml.CNOT(wires=[w1, w2])
        param_idx += 1
        
    # 2. Even Pairs (짝수 쌍): (1,2), (3,4), ...
    for i in range(1, n - 1, 2):
        w1, w2 = wires[i], wires[i+1]
        qml.CNOT(wires=[w1, w2])
        qml.RZ(params[param_idx], wires=w2)
        qml.CNOT(wires=[w1, w2])
        param_idx += 1

    return param_idx


class PennyLanePQCLayer(nn.Module):
    def __init__(self, num_qubits, reps=2):
        super(PennyLanePQCLayer, self).__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        num_conv_even = len(range(0, num_qubits - 1, 2))
        num_conv_odd  = len(range(1, num_qubits, 2)) 
        self.total_conv_params = (num_conv_even + num_conv_odd) * 3
        self.total_zz_params = num_qubits - 1
        self.total_mixing_params = 2 * num_qubits + 3 * num_qubits
        self.total_params_per_layer = self.total_conv_params + self.total_zz_params + self.total_mixing_params
        
        # 1. Device
        try:
            dev = qml.device("lightning.qubit", wires=num_qubits)
        except:
            dev = qml.device("default.qubit", wires=num_qubits)

        # 2. QNode (Quantum Circuit)
        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def quantum_circuit(inputs, weights):
            
            # (1) Data Encoding
            qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.)

            # (2) Ansatz
            for i in range(num_qubits):
                qml.H(wires=[i])
            
            for d in range(reps):
                # --- [Layer 1] Convolution (Flattened Params) ---
                current_layer_params = weights[d]
                idx = 0

                # --- Step 1: Convolution ---
                # Even Pairs
                for i in range(0, num_qubits-1, 2):
                    idx += ConvUnit(current_layer_params[idx:], wires=[i, (i + 1) % num_qubits])
                
                # Odd Pairs
                for i in range(1, num_qubits, 2):
                    idx += ConvUnit(current_layer_params[idx:], wires=[i, (i + 1) % num_qubits])
                

                idx += ZZInteractionLayer(current_layer_params[idx:], wires=range(num_qubits))

                # --- Step 3: Mixing --- 
                # 2 times mixing per layer
                idx += MixingBlock(current_layer_params[idx:], wires=range(num_qubits))
            
            # (3) Measurement (Expectation Value of Z for each qubit)
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.qnode = quantum_circuit

    def forward(self, x, weights):
        result_list = self.qnode(x, weights)
        return torch.stack(result_list, dim=-1)


# --- [3] Qiskit or PennyLane Hybrid Quantum U-Net ---

class QuantumUNet(nn.Module):
    """
    Hybrid Quantum-Classical U-Net (PennyLane Version)
    Encoder -> PennyLane PQC (Bottleneck) -> Decoder
    """
    def __init__(self, num_qubits, layers, T, init_variance, betas, activation=False, device='cuda', bottleneck_qubits=4, use_pooling=False):
        super(QuantumUNet, self).__init__()
        
        self.device = torch.device(device)
        self.T = T
        self.betas = betas
        self.init_variance = init_variance
        self.best_loss = float('inf')
        
        # Dimensions
        input_dim = 2 ** num_qubits          # ex: 256
        input_flat_dim = input_dim * 2       # ex: 512 (Real+Imag)
        hidden_dim = (2 ** num_qubits) * 2   # ex: 512
        self.pqc_input_dim = 2**bottleneck_qubits
        
        # --- [1] Encoder ---
        # Complex Input -> Latent Feature (Real)
        self.enc1 = ComplexLinear(input_dim, hidden_dim)
        self.pqc_input_norm = nn.LayerNorm(self.pqc_input_dim)
         
        self.enc2 = nn.Linear(hidden_dim, self.pqc_input_dim) # Output matches QNN input size

        # --- [2] PennyLane or Qiskit PQC Bottleneck ---

        #self.pqc = QiskitPQCLayer(num_qubits=bottleneck_qubits, reps=layers)
        self.pqc_layer = PennyLanePQCLayer(num_qubits=bottleneck_qubits, reps=layers)
        
        self.pqc_weights = nn.Parameter(
            torch.randn(
                T, layers, self.pqc_layer.total_params_per_layer, 
                device=self.device
            ) * init_variance
        )

        self.dec1 = nn.Linear(bottleneck_qubits + input_flat_dim, hidden_dim)
        self.act_dec1 = nn.ReLU()
        
        self.final = nn.Linear(hidden_dim, input_dim * 2)

    def forward(self, input, t=None):
        # input: (T, Batch, Dim) Complex Tensor
        T_steps, batch_size, dim = input.shape
        
        # 1. Reshape & Flatten
        x = input.reshape(T_steps * batch_size, dim)
        x_flat = torch.cat([x.real, x.imag], dim=-1) # (T*B, 2*Dim)

        # --- Encoder ---
        e1 = self.enc1(x)         # (T*B, hidden_dim) [Real]
        latent = self.enc2(e1)    # (T*B, bottleneck_qubits) [Real] - These are angles for QNN

        latent = self.pqc_input_norm(latent)
        if torch.isnan(latent).any():
            latent = torch.nan_to_num(latent, nan=0.0)

        # --- PQC Execution ---
        epsilon = 1e-8
        latent = latent + 1e-6
        latent_norm = torch.norm(latent, p=2, dim=-1, keepdim=True)
        latent = latent / (latent_norm + epsilon)

        # --- Time-dependent Parameter Selection ---
        if t is None:
            selected_weights = self.pqc_weights[0] 
        else:
            if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.ndim == 0):
                idx = int(t) - 1
                idx = max(0, min(idx, self.T - 1))
                selected_weights = self.pqc_weights[idx]
            else:
                idx = int(t[0]) - 1
                idx = max(0, min(idx, self.T - 1))
                selected_weights = self.pqc_weights[idx]

        pqc_out = self.pqc_layer(latent, selected_weights)


        # --- Decoder ---
        d1_input = torch.cat([pqc_out, x_flat], dim=1)
        
        d1 = self.act_dec1(self.dec1(d1_input))
        out_feats = self.final(d1)
        
        # Output Reconstruction
        out_real = out_feats[:, :dim]
        out_imag = out_feats[:, dim:]
        
        # Normalize
        out = torch.complex(out_real, out_imag)
        
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + epsilon)
        
        return out.reshape(T_steps, batch_size, dim)

    # --- Utils for Compatibility with Trainer ---
    
    def save_params(self, directory, epoch=None, best=False):
        """Saves the entire model (including QNN weights)"""
        if best:
            filename = 'best_model.pt'
        elif epoch is not None:
            filename = f'epoch{epoch}_model.pt'
        else:
            filename = 'current_model.pt'
        
        save_path = f'{directory}/{filename}'
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'best_loss': self.best_loss
        }, save_path)

    def load_current_params(self, directory, epoch=None, noise=None):
        filename = f'epoch{epoch}_model.pt' if epoch is not None else 'current_model.pt'
        self._load_params_from_file(f'{directory}/{filename}')

    def load_best_params(self, directory, noise=None):
        self._load_params_from_file(f'{directory}/best_model.pt')

    def _load_params_from_file(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print(f"Loaded model from {path}")
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, skipping...")
        except Exception as e:
            print(f"Error loading model: {e}")

    def update_best_params(self, directory, losses):
        if isinstance(losses, (list, tuple)):
            current_loss = sum(losses)
        elif isinstance(losses, torch.Tensor):
            current_loss = losses.sum().item()
        else:
            current_loss = losses

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_params(directory, best=True)
            
    def get_pqc_params(self):
        return list(self.parameters())

    def get_mlp_params(self):
        return []