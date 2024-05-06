import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor

def plot_eeg(original_data, reconstructed_data, sampling_rate, title='EEG Data', channels=None):
    if channels is None:
        channels = [f'Channel {i+1}' for i in range(original_data.shape[0])]
    time = np.arange(original_data.shape[1]) / sampling_rate
    fig, axes = plt.subplots(len(channels), 2, figsize=(24, 10), sharex=True, sharey=True)
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax[0].plot(time, original_data[i, :], label=f'Original {channels[i]}', linewidth=1)
        ax[0].set_ylabel('Amplitude')
        ax[0].legend(loc='upper right')
        ax[1].plot(time, reconstructed_data[i, :], label=f'Reconstructed {channels[i]}', linewidth=1)
        ax[1].legend(loc='upper right')
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


data = pd.read_csv('small_dataset.csv')
X = data.iloc[:, :-1].values # Last col is the target
X = X.reshape((X.shape[0], 14, 1920)) # 26880 rows -> 14 channels, 15 secs x 128hz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

channel_means = X.mean(axis=(0, 2), keepdims=True)
channel_stds = X.std(axis=(0, 2), keepdims=True)
X = (X - channel_means) / channel_stds
X = Tensor(X).to(device)


class VQVAE(nn.Module):
    def __init__(self, input_channels, hidden_dims, num_codex_entries):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
        )
        self.codex = nn.Embedding(num_codex_entries, 240)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[2], hidden_dims[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dims[1], hidden_dims[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dims[1], input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.uniform_(self.codex.weight, -0.1, 0.1)

    def forward(self, x):
        encoded = self.encoder(x)
        flat_encoded = encoded.view(-1, encoded.size(-1))
        distances = torch.sum((flat_encoded[:, None] - self.codex.weight) ** 2, -1)
        nearest_indices = torch.argmin(distances, dim=1)
        quantized = self.codex(nearest_indices).view(encoded.size())
        reconstructed_x = self.decoder(quantized)
        return reconstructed_x, encoded, quantized


input_channels = 14  # Number of EEG channels
hidden_dims = [32, 64, 240]
num_codex_entries = 1024  # Codex entries. 256-2048 should be fine. Crank it up if you've some A100s nearby.

#X = Tensor(X)

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X).to(device)
#dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # You can try batch if you'd like, but you'll have to modify the training loop a bit

model = VQVAE(input_channels, hidden_dims, num_codex_entries).to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-3).to(device)
reconstruction_loss_fn = nn.MSELoss()

def compute_vq_loss(encoded, quantized):
    # Commitment loss to ensure the encoder's outputs match the quantized vectors
    commitment_loss = torch.mean((encoded.detach() - quantized) ** 2)
    
    # Codebook loss to update the embeddings to better match the encoder's outputs
    codebook_loss = torch.mean((encoded - quantized.detach()) ** 2)
    
    beta = 0.25 
    vq_loss = codebook_loss + beta * commitment_loss
    return vq_loss


num_epochs = 100

for epoch in range(num_epochs): # Training loop
    model.train()
    running_loss = 0.0
    for batch in dataset.tensors[0]:
        x = batch
        x = x.unsqueeze(0) # (14, 1920) -> (1, 14, 1920)
        optimizer.zero_grad()

        reconstructed_x, encoded, quantized = model(x)
        reconstruction_loss = reconstruction_loss_fn(reconstructed_x, x)
        
        vq_loss = compute_vq_loss(encoded, quantized)
        
        #loss = reconstruction_loss + vq_loss # Need to optimize the model so the VQ Loss is no longer a bottlenecks for the whole loss
        loss = reconstruction_loss

        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataset.tensors[0])
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')


reconstructed_x,_ ,_ = model(X)
reconstructed_x = reconstructed_x.detach().numpy()
plot_eeg(X[-1], reconstructed_x[-1], 128, 'EEG Data Comparison')
