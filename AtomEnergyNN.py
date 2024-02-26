import json
import numpy as np
from dscribe.descriptors import ACSF
from pymatgen.core import Structure
from ase import Atoms
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.utils.rnn as rnn_utils
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
S_min = -1
S_max = 1
acsf_descriptors = []
target_energies = []

def precondition_symmetry_functions(symmetry_functions, S_min, S_max):
    # Step 1: Shifting the Mean to Zero
    mean_values = np.mean(symmetry_functions, axis=0)
    shifted_symmetry_functions = symmetry_functions - mean_values

    # Step 2: Rescaling to a Predefined Interval
    min_values = np.min(shifted_symmetry_functions, axis=0)
    max_values = np.max(shifted_symmetry_functions, axis=0)
    
    # Ensure no division by zero
    max_values[max_values == min_values] += 1e-12
    
    rescaled_symmetry_functions = (shifted_symmetry_functions - min_values) / (max_values - min_values) * (S_max - S_min) + S_min
    
    return rescaled_symmetry_functions


with open('position_force_train_all.json', 'r') as f:
    p_f = json.loads(f.read())

acsf = ACSF(
    species=["B"],
    r_cut=4.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    periodic=True
)

# Initialize empty lists to store ACSF descriptors and target energies
acsf_descriptors = []
target_energies = []

# for i in range(len(p_f)):
for i in range(1000):
    pmg_structure = Structure.from_dict(p_f[i]['structure'])
    energy = p_f[i]['energy']
    forces = p_f[i]['forces']
    bc = p_f[i]['bc']
    pbc = [0, 0, 0] if 'free' in bc else [1,1,1]
    ase_structure = Atoms(pmg_structure.composition.formula, positions=pmg_structure.cart_coords, cell=pmg_structure.lattice.matrix, pbc=pbc)
    acsf_value = acsf.create(ase_structure)
    acsf_descriptors.append(precondition_symmetry_functions(acsf_value, S_min, S_max))
#     print(precondition_symmetry_functions(acsf_value, S_min, S_max))
    target_energies.append(energy)

packed_acsf_tensor = rnn_utils.pack_sequence([torch.tensor(seq) for seq in acsf_descriptors], enforce_sorted=False)

# Use the packed sequence directly or convert it to a padded sequence if needed
# Example: Convert packed sequence to a padded sequence
padded_acsf_tensor, _ = rnn_utils.pad_packed_sequence(packed_acsf_tensor, batch_first=True, padding_value=0.0)

# Assuming target_energies is a NumPy array containing the target total energies for each structure
target_energies = torch.tensor(target_energies, dtype=torch.float32)

# Combine ACSF descriptors and target energies into a PyTorch Dataset
dataset = TensorDataset(padded_acsf_tensor, target_energies)

# Split the dataset into training and validation sets (80% - 20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation sets
batch_size = 500
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define your neural network architecture
class AtomEnergyNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(AtomEnergyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1, dtype=torch.double)
        self.relu1 = nn.ReLU()                                                         #self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2, dtype=torch.double)
        self.relu2= nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, output_size, dtype=torch.double)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Set parameters
input_size = 8  # Number of features in ACSF descriptor for each atom
hidden_size_1 = 16  # Number of neurons in the hidden layer 1
hidden_size_2 = 16 # Number of neurons in the hidden layer 2
output_size = 1  # Single output for each atom's energy

# Create the neural network
model = AtomEnergyNN(input_size, hidden_size_1, hidden_size_2, output_size)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()  # Mean Squared Error loss
# optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define your loss function, for example:
loss_function = torch.nn.MSELoss()
requires_grad = True

# Define batch size
batch_size = 100  # Adjust as needed

# Define number of epochs
num_epochs = 10  # Adjust as needed

# Training loop with epochs and batch training
for epoch in range(num_epochs):
    batch_no = 0
    # Iterate over the data in batches within each epoch
    for batch_start in range(0, len(padded_acsf_tensor), batch_size):
        # Zero the gradients
        optimizer.zero_grad()
        batch_no += 1
        batch_end = min(batch_start + batch_size, len(padded_acsf_tensor))
        batch_acsf_tensor = padded_acsf_tensor[batch_start:batch_end]

        # Compute total energy for each structure in the batch
        total_energies = []
        target_total_energies = []
        for struct_acsf_tensor, target_energy in zip(batch_acsf_tensor, target_energies[batch_start:batch_end]):
            struct_acsf_tensor = struct_acsf_tensor[struct_acsf_tensor.any(dim=1)]
            total_energy = model(struct_acsf_tensor).sum().item()
            total_energies.append(total_energy)
            target_total_energies.append(target_energy.item())

            # Print predicted and actual energies for each structure
            print(f"Epoch {epoch + 1}/{num_epochs} Batch {batch_no}")
            print('Predicted Energy and actual energy:', total_energy, target_energy.item())

        # Compute loss for the batch
        predicted_energies = torch.tensor(total_energies, dtype=torch.float32, requires_grad = True)
        target_energies_batch = torch.tensor(target_total_energies, dtype=torch.float32)
        loss = criterion(predicted_energies, target_energies_batch)
        print('Loss:', loss.item())

        # Check if gradients are being computed for fc1.weight
        for name, param in model.named_parameters():
            if name == 'fc2.weight':
                print(name, param.data)  # Ensure that fc1.weight requires gradients

#         # Zero the gradients
#         optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()

        # Update the weights
        optimizer.step()