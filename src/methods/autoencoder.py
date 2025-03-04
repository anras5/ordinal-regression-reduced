import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset


# Define the Autoencoder model.
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        Autoencoder with configurable latent space dimensionality.
        """
        super(AutoEncoder, self).__init__()
        # Encoder: input -> hidden -> latent
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        # Decoder: latent -> hidden -> reconstruction of input
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def compute_dominance_penalty(x, z, margin=0.1):
    """
    Computes the pairwise dominance penalty.

    For every pair (i, j) in the batch, if sample i dominates sample j in the input space
    (i.e., all features of x[i] are >= corresponding features of x[j] with at least one strictly greater),
    then add a hinge loss term to ensure that z[i] is at least `margin` higher than z[j] for each latent dimension.

    The hinge loss per latent dimension is:
        loss = max(0, z[j] - z[i] + margin)
    """
    penalty = 0.0
    batch_size = x.size(0)
    # Loop over all pairs in the batch
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            # Check dominance: sample i dominates sample j.
            if torch.all(x[i] >= x[j]) and torch.any(x[i] > x[j]):
                diff = z[j] - z[i] + margin
                penalty += F.relu(diff).sum()
    return penalty


class DominanceAutoEncoder(BaseEstimator, TransformerMixin):
    """
    An sklearn-like autoencoder that accepts a pandas DataFrame,
    trains with a dominance penalty, and provides a fit_transform method.

    This implementation includes:
      - A random_state parameter to set seeds in the fit() method.
      - A verbose mode to print training loss.
      - Compatibility with sklearn's Pipeline.
    """

    def __init__(
        self,
        latent_dim=2,
        num_epochs=10,
        lambda_rank=1.0,
        margin=0.1,
        lr=1e-3,
        batch_size=32,
        random_state=None,
        device=None,
        verbose=False,
    ):
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.model = None
        self.input_dim = None

    def _set_random_seed(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _build_model(self, input_dim):
        self.input_dim = input_dim
        self.model = AutoEncoder(input_dim=input_dim, latent_dim=self.latent_dim).to(self.device)

    def fit(self, X, y=None):
        """
        Fit the autoencoder on a pandas DataFrame X.

        Parameters:
            X (numpy array): Input data.
            y: Ignored, exists for compatibility with sklearn's Pipeline.

        Returns:
            self
        """
        # Set random seeds for reproducibility.
        self._set_random_seed()

        # Convert DataFrame to tensor (float32) and build DataLoader.
        X_np = X.astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(X_np))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.input_dim is None:
            self._build_model(input_dim=X_np.shape[1])

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss()

        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                x_hat, z = self.model(x)
                loss_recon = mse_loss(x_hat, x)
                loss_dom = compute_dominance_penalty(x, z, margin=self.margin)
                loss = loss_recon + self.lambda_rank * loss_dom
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < 0.001:
                break
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame X into the latent space.

        Parameters:
            X (pandas.DataFrame): Input data.

        Returns:
            numpy.ndarray: The latent representations.
        """
        self.model.eval()
        X_np = X.astype(np.float32)
        X_tensor = torch.from_numpy(X_np).to(self.device)
        with torch.no_grad():
            _, z = self.model(X_tensor)
        return z.cpu().numpy()

    def fit_transform(self, X, y=None):
        """
        Fit the model on X and return the latent representations.

        Parameters:
            X (pandas.DataFrame): Input data.
            y: Ignored, exists for compatibility.

        Returns:
            numpy.ndarray: The latent representations.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        """
        Return parameters for this estimator.
        """
        return {
            "latent_dim": self.latent_dim,
            "num_epochs": self.num_epochs,
            "lambda_rank": self.lambda_rank,
            "margin": self.margin,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "device": self.device,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
