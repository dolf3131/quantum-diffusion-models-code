# src.models.custom_classes.py

import torch
import torch.nn as nn

"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""

def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size([a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1]])
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

def CNOT_to_layer(position, num_qubits, device):
    '''
    Returns the matrix obtained by taking the tensor product of identities with 
    a CNOT matrix at the specified position.
    '''
    # Create the identity matrix
    identity = torch.eye(2, device=device)

    # Create the CNOT matrix
    cnot = torch.tensor(([1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,1,0]), device=device)

    # Create the list of matrices to be tensored together
    matrices = [identity] * num_qubits
    matrices[position] = cnot
    del matrices[position+1]
    out = torch.eye(1, device = device)

    # Tensor the matrices together
    for matrix in matrices:
        out = torch.kron(out, matrix)

    # out shape: (2^num_qubits, 2^num_qubits)
    return out

class EntanglingBlock(nn.Module):
    '''
    Returns the unitary matrix corresponding to the entangling block
    of the Strongly Entangling Circuit (SEC) ansatz.
    '''
    def __init__(self, num_qubits, device):
        super(EntanglingBlock, self).__init__()
        self.device = device
        self.num_qubits = num_qubits

        layers = []
        for i in range(self.num_qubits - 1):
            layers.append(CNOT_to_layer(i, self.num_qubits, device=self.device))

        # return matrix product of layers from last to first
        self.prepared_entangling_block_matrix = torch.eye(2**self.num_qubits, device=self.device)
        for layer in reversed(layers):
            self.prepared_entangling_block_matrix = torch.matmul(layer, self.prepared_entangling_block_matrix)

    def forward(self): return self.prepared_entangling_block_matrix
    
class R3Matrix(nn.Module):
    '''
    Takes a tensor of angles and returns the unitary matrix corresponding to the R3 gate
    input is expected to be a tensor of shape (batch_size, 3)
    '''
    def __init__(self):
        super(R3Matrix, self).__init__()

    def forward(self, angles):

        # angles shape: (T, 3)

        # Compute the R3 matrix dynamically based on omega, theta and phi
        omega, theta, phi = angles[:,0], angles[:,1], angles[:,2]

        m11 = torch.exp(-1j * (phi+omega)/2) * torch.cos(theta/2)
        m12 = -torch.exp(1j * (phi-omega)/2) * torch.sin(theta/2)
        m21 = torch.exp(-1j * (phi-omega)/2) * torch.sin(theta/2)
        m22 = torch.exp(1j * (phi+omega)/2) * torch.cos(theta/2)

        # shape: (T, 1, 2)
        upper_row = torch.cat([m11.view(-1, 1, 1), m12.view(-1, 1, 1)], dim=-1)

        # shape: (T, 1, 2)
        lower_row = torch.cat([m21.view(-1, 1, 1), m22.view(-1, 1, 1)], dim=-1)

        # shape: (T, 2, 2)
        r3_matrix = torch.cat([upper_row, lower_row], dim=-2)

        return r3_matrix

class RotationR3Block(nn.Module):

    def __init__(self, num_qubits, device):
        super(RotationR3Block, self).__init__()
        self.num_qubits = num_qubits
        self.device = device

        self.rotation_matrix = R3Matrix()

    def forward(self, angles):

        # angles shape: (T, num_qubts, 3)

        out_matrix = torch.eye(1, device=self.device)

        for qubit in range(self.num_qubits):
            rotation_matrix = self.rotation_matrix(angles[:, qubit, :])
            out_matrix = kron(out_matrix, rotation_matrix)

        # out_matrix shape: (T, 2^num_qubits, 2^num_qubits)

        return out_matrix
    
class UnitaryR3Ansatz(nn.Module):
    '''
    - init: takes the number of qubits and layers and
    builds the unitary matrix corresponding to the SEC ansatz
    angles are expected to be a tensor of shape (batch_size, num_layers, num_qubits, 3)
    - forward: takes angles and input and returns the output of the circuit
    '''
    def __init__(self, num_qubits, layers, device):
        super(UnitaryR3Ansatz, self).__init__()
        self.num_qubits = num_qubits
        self.layers = layers
        self.device = device

        self.entangling_block = EntanglingBlock(num_qubits, device)
        self.rotation_block = RotationR3Block(num_qubits, device)

    def forward(self, input, params):

        # angles is a tensor of shape (batch_size, num_layers, num_qubits, 3)
        # return matrix product of entangling block and rotation block
        batch_size = int(input.shape[0]/params.shape[0])
        copies = params.shape[0]

        # shape: (T, 2^num_qubits, 2^num_qubits)
        matrix = torch.eye(2**self.num_qubits, device=self.device).unsqueeze(0).expand(copies, -1, -1).to(torch.complex64)
        
        # expand entangling block to copies
        entangling_block = self.entangling_block().unsqueeze(0).expand(copies, -1, -1).to(torch.complex64)
        
        for layer in range(self.layers):

            matrix = torch.bmm(self.rotation_block(params[:,layer,:,:]), matrix)
            matrix = torch.bmm(entangling_block, matrix)
        
        # copy matrix to match the input shape
        matrix = torch.repeat_interleave(matrix, batch_size, dim=0)

        # return the output of the circuit
        output = torch.bmm(matrix, input.unsqueeze(-1)).squeeze()

        return output
    
class IdentityUnitaryR3Ansatz(nn.Module):
    '''
    - init: takes the number of qubits and layers and
    builds the unitary matrix corresponding to the SEC ansatz
    angles are expected to be a tensor of shape (batch_size, num_layers, num_qubits, 3)
    - forward: takes angles and input and returns the output of the circuit
    '''
    def __init__(self, num_qubits, layers, device):
        super(IdentityUnitaryR3Ansatz, self).__init__()
        self.num_qubits = num_qubits
        self.layers = layers
        self.device = device

        # if layers is odd, reject
        if layers % 2 == 1:
            raise ValueError('The number of layers must be even')

        self.entangling_block = EntanglingBlock(num_qubits, device)
        self.rotation_block = RotationR3Block(num_qubits, device)
        self.reversed_entangling_block = torch.conj(self.entangling_block().T)

    def forward(self, input, params):

        # input shape: (T*BS, 2^num_qubits)
        # params shape: (T, num_layers, num_qubits, 3)

        batch_size = int(input.shape[0]/params.shape[0])
        T = params.shape[0]

        # shape: (T, 2^num_qubits, 2^num_qubits)
        matrix = torch.eye(2**self.num_qubits, device=input.device).unsqueeze(0).expand(T, -1, -1).to(torch.complex64)
        
        # shape: (T, 2^num_qubits, 2^num_qubits)
        entangling_block = self.entangling_block().unsqueeze(0).expand(T, -1, -1).to(torch.complex64)
        reversed_entangling_block = self.reversed_entangling_block.unsqueeze(0).expand(T, -1, -1).to(torch.complex64)
        
        for layer in range(self.layers):

            matrix = torch.bmm(self.rotation_block(params[:,layer,:,:]), matrix)
            
            # apply entangling block if layer is even, otherwise apply the reversed entangling block
            if layer % 2 == 0: matrix = torch.bmm(entangling_block, matrix)
            else: matrix = torch.bmm(reversed_entangling_block, matrix)
        
        # shape: (T^BS, 2^num_qubits, 2^num_qubits)
        matrix = torch.repeat_interleave(matrix, batch_size, dim=0)

        # shape: (T^BS, 2^num_qubits)
        output = torch.bmm(matrix, input.unsqueeze(-1)).squeeze()

        return output
        
##############################################################################################################
    
class R3Matrix_batch(nn.Module):
    """
    Compute batched R3 single-qubit rotation matrices (T, batch_size, 2, 2) from angles (omega, theta, phi).
    """

    def __init__(self):
        super(R3Matrix_batch, self).__init__()

    def forward(self, angles):

        # angles shape: (T, batch_size, 3)

        omega, theta, phi = angles[:,:,0], angles[:,:,1], angles[:,:,2]

        # Complex exponential factors
        exp_factor_1 = torch.exp(-1j * (phi + omega) / 2)
        exp_factor_2 = torch.exp(1j * (phi - omega) / 2)
        exp_factor_3 = torch.exp(-1j * (phi - omega) / 2)
        exp_factor_4 = torch.exp(1j * (phi + omega) / 2)

        cos_term = torch.cos(theta / 2)
        sin_term = torch.sin(theta / 2)

        # Compute matrix elements
        m11 = exp_factor_1 * cos_term
        m12 = -exp_factor_2 * sin_term
        m21 = exp_factor_3 * sin_term
        m22 = exp_factor_4 * cos_term

        # Reshape to (T, batch_size, 1, 1) so we can concatenate easily
        m11 = m11.unsqueeze(-1).unsqueeze(-1)
        m12 = m12.unsqueeze(-1).unsqueeze(-1)
        m21 = m21.unsqueeze(-1).unsqueeze(-1)
        m22 = m22.unsqueeze(-1).unsqueeze(-1)

        # shape: (T, batch_size, 1, 2)
        upper_row = torch.cat([m11, m12], dim=-1)
        # shape: (T, batch_size, 1, 2)
        lower_row = torch.cat([m21, m22], dim=-1)

        # shape: (T, batch_size, 2, 2)
        r3_matrix = torch.cat([upper_row, lower_row], dim=-2)

        return r3_matrix


class RotationR3Block_batch(nn.Module):
    """
    Computes the full rotational block for multiple qubits by taking the tensor product of individual R3 gates.

    - Input angles: (T, batch_size, num_qubits, 3)
    - For each qubit, we get an R3 unitary. Then we take the tensor product of all R3 gates to get a full multi-qubit unitary.
    """

    def __init__(self, num_qubits, device):
        super(RotationR3Block_batch, self).__init__()
        self.num_qubits = num_qubits
        self.device = device
        self.rotation_matrix = R3Matrix_batch()

    def forward(self, angles):

        # angles shape: (T, BS, num_qubits, 3)

        # for each qubit, compute its R3 matrix
        rotation_matrices = []
        for q in range(self.num_qubits):

            # shape: (T, BS, 3)
            qubit_angles = angles[:,:,q,:]

            # shape: (T, BS, 2^num_qubits, 2^num_qubits)
            rotation_matrix = self.rotation_matrix(qubit_angles)

            rotation_matrices.append(rotation_matrix)

        # Compute the tensor product across all qubits
        # shape: (1, 1)
        out = torch.eye(1, device=self.device).to(torch.complex64)
        for mat in rotation_matrices:
            if out.dim() == 2:
                out = out.unsqueeze(0).unsqueeze(0)
                out = out.expand(mat.shape[0], mat.shape[1], 1, 1)

            # outer product
            out = kron(out, mat)

        # out
        # shape: (T, BS, 2^num_qubits, 2^num_qubits)

        return out

class IdentityUnitaryR3Ansatz_batch(nn.Module):
    """
    Constructs a parameterized unitary ansatz composed of alternating rotation (R3) blocks and entangling blocks.
    The pattern is something like: 
        For each layer:
            Apply a rotation block parameterized by angles
            Then apply either the entangling block or its conjugate transpose, alternating by layer.

    Input/Output shape:
    - Input state: (T*batch_size, 2^(num_qubits))
    - Params: (T, batch_size, num_layers, num_qubits, 3)

    Returns:
    - Transformed state after applying all layers.

    Note: The number of layers must be even.
    """

    def __init__(self, num_qubits, layers, batch_size, device):
        super(IdentityUnitaryR3Ansatz_batch, self).__init__()
        if layers % 2 == 1:
            raise ValueError('The number of layers must be even')
        
        self.num_qubits = num_qubits
        self.layers = layers
        self.device = device
        self.batch_size = batch_size

        self.entangling_block = EntanglingBlock(num_qubits, device)
        self.rotation_block = RotationR3Block_batch(num_qubits, device=device)

        # Conjugate transpose of the entangling block
        E = self.entangling_block().to(device)  # (2^N, 2^N)
        self.reversed_entangling_block = torch.conj(E.T)  # (2^N, 2^N)

    def forward(self, input_state, params):
        """
        input_state: (T*batch_size, 2^(num_qubits))
        params: (T, batch_size, num_layers, num_qubits, 3)

        Applies [Rotation -> Entangling or Inverse Entangling] for each layer.
        """

        T = params.shape[0]
        batch_size = params.shape[1]

        # construct a final unitary by sequential multiplication
        # shape: (2^num_qubits, 2^num_qubits)
        identity = torch.eye(2**self.num_qubits, device=self.device).to(torch.complex64)
        
        # shape: (T*BS, 2^num_qubits, 2^num_qubits)
        matrix = identity.unsqueeze(0).expand(T * batch_size, -1, -1)

        # Pre-expand entangling and reversed blocks
        # shape: (T*BS, 2^num_qubits, 2^num_qubits)
        entangling_block = self.entangling_block().unsqueeze(0).expand(T*batch_size, -1, -1).to(torch.complex64)
        reversed_ent_block = self.reversed_entangling_block.unsqueeze(0).expand(T*batch_size, -1, -1).to(torch.complex64)

        # Apply each layer
        for layer_index in range(self.layers):

            # shape: (T, BS, num_qubits, 3)
            layer_params = params[:,:,layer_index,:,:]  # extract params for this layer

            # Compute rotation block for these params
            # shape: (T*batch_size, 2^num_qubits, 2^num_qubits)
            rotation_block = self.rotation_block(layer_params).reshape(T*batch_size, 2**self.num_qubits, 2**self.num_qubits)

            # matrix = R * matrix
            # shape: (T*BS, 2^num_qubits, 2^num_qubits)
            matrix = torch.bmm(rotation_block, matrix)

            # Apply entangling or reversed entangling block
            if layer_index % 2 == 0:
                # even layer -> entangling_block
                matrix = torch.bmm(entangling_block, matrix)
            else:
                # odd layer -> reversed entangling_block
                matrix = torch.bmm(reversed_ent_block, matrix)

        # Apply the resulting unitary to the input state
        # shape: (T*BS, 2^num_qubits)
        output = torch.bmm(matrix, input_state.unsqueeze(-1)).squeeze(-1)

        return output
