import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderModel:
    def __init__(
        self,
        hidden_dim=64,
        encoded_dim=3,
        learning_rate=1e-3,
        epochs=50,
        batch_size=32,
        seed=42,
    ):
        self.hidden_dim = hidden_dim
        self.encoded_dim = encoded_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.autoencoder = None
        self.optimizer = None
        self.criterion = None
        self._set_seed()

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit_transform(self, data):
        # Automatically detect input dimensions
        input_dim = data.shape[1]

        # Initialize the autoencoder with the detected input dimensions
        self.autoencoder = Autoencoder(input_dim, self.hidden_dim, self.encoded_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)

        # Convert DataFrame to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        self.autoencoder.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                inputs, _ = batch
                outputs = self.autoencoder(inputs)
                loss = self.criterion(outputs, inputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}')

        # Transforming
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_data = self.autoencoder.encode(data_tensor)
        return encoded_data.numpy()

    def test(self, data):
        # Convert DataFrame to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        self.autoencoder.eval()
        with torch.no_grad():
            decoded_data = self.autoencoder(data_tensor)
        return decoded_data.numpy()


if __name__ == "__main__":
    import numpy as np

    # Example data with 10 attributes
    data = np.array(
        [
            [-1.8342e04, -3.0700e01, -3.7200e01, 2.3300e00, 3.0000e00],
            [-1.5335e04, -3.0200e01, -4.1600e01, 2.0000e00, 2.5000e00],
            [-1.6973e04, -2.9000e01, -3.4900e01, 2.6600e00, 2.5000e00],
            [-1.5460e04, -3.0400e01, -3.5800e01, 1.6600e00, 1.5000e00],
            [-1.5131e04, -2.9700e01, -3.5600e01, 1.6600e00, 1.7500e00],
            [-1.3841e04, -3.0800e01, -3.6500e01, 1.3300e00, 2.0000e00],
            [-1.8971e04, -2.8000e01, -3.5600e01, 2.3300e00, 2.0000e00],
            [-1.8319e04, -2.8900e01, -3.5300e01, 1.6600e00, 2.0000e00],
            [-1.9800e04, -2.9400e01, -3.4700e01, 2.0000e00, 1.7500e00],
            [-1.6966e04, -3.0000e01, -3.7700e01, 2.3300e00, 3.2500e00],
            [-1.7537e04, -2.8300e01, -3.4800e01, 2.3300e00, 2.7500e00],
            [-1.5980e04, -2.9600e01, -3.5300e01, 2.3300e00, 2.7500e00],
            [-1.7219e04, -3.0200e01, -3.6900e01, 1.6600e00, 1.2500e00],
            [-2.1334e04, -2.8900e01, -3.6700e01, 2.0000e00, 2.2500e00],
        ]
    )

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Initialize AutoencoderModel with default parameters (hidden_dim=64, encoded_dim=3)
    autoencoder_model = AutoencoderModel(epochs=200, batch_size=16)

    # Fit and transform the data
    reduced_data = autoencoder_model.fit_transform(data)
    # print("Reduced data shape:", reduced_data.shape)
    # print(reduced_data)

    # Original data
    print("Original data shape: ", data.shape)
    print(data)

    # Decoded df
    decoded_data = autoencoder_model.test(data)
    print("Decoded data shape:", decoded_data.shape)
    print(decoded_data)
