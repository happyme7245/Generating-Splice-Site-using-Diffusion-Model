import os
import torch
import numpy as np
import pandas as pd
from data_utils import Utils
from dna_dataset import DNADataset
from models import SimpleUNet, DDPM, UNet1D
from torch.utils.data import DataLoader
from one_hot import OneHotEncodeTransform
from sklearn.model_selection import train_test_split


# Hyperparameters
epochs = 80
loss_type = 'huber' # option -> 'l1'. 'l2', 'huber'
betas = np.linspace(1e-4, 0.02, 1000).tolist()
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
print(f"GPU operation: {device}")

# Model, optimizer, and loss function
input_dim = 4  # DNA sequence one-hot encoded dimension
unet = SimpleUNet(input_dim, 32).to(device)
ddpm = DDPM(betas, unet).to(device)
optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)


path = os.path.join('REAL_dataset', 'clean_acceptors.pos')  # dataset pathway

with open(path, 'r') as file:
    lines = file.readlines()
    
df1 = pd.DataFrame(lines, columns=['DNA_sequence'])
df = df1.replace('\n', '', regex=True)  # Remove newline characters from the entire DataFrame

# Extract sequences and labels
sequences = df['DNA_sequence'].tolist()

# Split the dataset
train_seqs, temp_seqs = train_test_split(sequences, test_size=0.2, random_state=42)
valid_seqs, test_seqs = train_test_split(temp_seqs, test_size=0.5, random_state=42)

# Define the transform
onehot_transform = OneHotEncodeTransform()

# Create datasets
train_dataset = DNADataset(train_seqs, transform=onehot_transform)
valid_dataset = DNADataset(valid_seqs, transform=onehot_transform)
test_dataset = DNADataset(test_seqs, transform=onehot_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training and Validation Loop
for epoch in range(epochs):
    # Training Phase
    ddpm.train()  # Set model to training mode
    train_loss = 0.0
    for step, x in enumerate(train_loader):
        x = x.clone().detach().requires_grad_(True).to(device)
        t = torch.randint(0, len(betas), (x.size(0),), device=device).long()

        optimizer.zero_grad()
        loss = ddpm.p_losses(x, t, loss_type=loss_type)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation Phase
    ddpm.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for x in valid_loader:
            x = x.to(device)
            t = torch.randint(0, len(betas), (x.size(0),), device=device).long()
            
            loss = ddpm.p_losses(x, t, loss_type=loss_type)
            val_loss += loss.item() * x.size(0)

    val_loss /= len(valid_loader.dataset)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

# write generated DNA sequences
write_path = os.path.join('SYNTHETIC_dataset', f'synthetic_dataset.txt')
util = Utils(ddpm, device)
generated_sequences = util.seq_to_file(num_sequences=len(train_dataset), seq_length=402, file=write_path)

# Save the model
torch.save({'ddpm_state_dict': ddpm.state_dict(),
            'unet_state_dict': ddpm.unet.state_dict()}, 
            f'saved_models/ddpm_{count}.pth')

ddpm_model_path = os.path.join('saved_models', f'ddpm_{count}.pth')

# Load models
checkpoint = torch.load(ddpm_model_path)

# Load DDPM state dict
ddpm.load_state_dict(checkpoint['ddpm_state_dict'])

# Make test loader if using different dataset for testing
output_file_path = os.path.join('SYNTHETIC_dataset', 'SPC_acceptor_TRTR_pos.txt')

with open(output_file_path, 'r') as file:
    lines = file.readlines()
    
df1 = pd.DataFrame(lines, columns=['DNA_sequence'])
df = df1.replace('\n', '', regex=True)  # Remove newline characters from the entire DataFrame

# Extract sequences and labels
sequences = df['DNA_sequence'].tolist()

test_dataset = DNADataset(sequences, transform=onehot_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Generate predictions
predictions = util.generate_predictions(test_loader, output_file_path)
