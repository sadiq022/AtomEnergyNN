import json
import numpy as np
from dscribe.descriptors import ACSF
from pymatgen.core import Structure
from ase import Atoms
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.utils.rnn as rnn_utils
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import matplotlib.pyplot as plt
import time

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
for i in range(10000):
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
requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.001)     #SGD optimizer

# Define batch size
batch_size = 100  # Adjust as needed

# Define number of epochs
num_epochs = 50  # Adjust as needed
# Initialize list to store losses
epoch_losses = []
epoch_times = []

# Training loop with epochs and batch training
for epoch in range(num_epochs):
    epoch_loss = 0
    batch_no = 0
    start_time = time.time()
    # Iterate over the data in batches within each epoch
    for batch_start in range(0, len(padded_acsf_tensor), batch_size):
        batch_no += 1
        loss = 0
        batch_end = min(batch_start + batch_size, len(padded_acsf_tensor))
        batch_acsf_tensor = padded_acsf_tensor[batch_start:batch_end]

        for struct_acsf_tensor, target_energy in zip(batch_acsf_tensor, target_energies[batch_start:batch_end]):
            total_energy = 0
            struct_acsf_tensor = struct_acsf_tensor[struct_acsf_tensor.any(dim=1)]
            total_energy = model(struct_acsf_tensor).sum()
            loss += (target_energy - total_energy)**2

            # Print predicted and actual energies for each structure
#             print(f"Epoch {epoch + 1}/{num_epochs} Batch {batch_no}")
#             print('Predicted Energy and actual energy:', total_energy, target_energy)
        
        loss = loss/(2*batch_size)
#         print('Loss:', loss)

        # Zero the gradients
        optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()

        # Update the weights
        optimizer.step()
        epoch_loss += loss.item()
        
    end_time = time.time()  # Record end time for epoch
    epoch_time = end_time - start_time  # Calculate epoch time
    # Save the epoch loss
    epoch_losses.append(epoch_loss)
    epoch_times.append(epoch_time)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_loss}, Time: {epoch_time:.2f} seconds")

# Plot the losses
plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# Plot the epoch times
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_times, color='tab:red', label='Epoch Time')
plt.xlabel('Epochs')
plt.ylabel('Time (seconds)')
plt.title('Epoch Time Over Epochs')
plt.legend()
plt.show()
