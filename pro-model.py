from tqdm import tqdm
from typing import *
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

# adjust:
# Layer sizes
# Dropout rate (currently 0.2)
# Learning rate (currently 0.001)
# Add or remove layers
# Batch size during training
# Threshold value for binary prediction (default 0.5)
class ProModel(nn.Module):
    def __init__(self, embedding_dim=1280, dropout=0.2):
        super(ProModel, self).__init__()

        # First dense layer
        self._dense1 = nn.Linear(embedding_dim, 512)
        self._batch_norm1 = nn.BatchNorm1d(512)

        # Second dense layer
        self._dense2 = nn.Linear(512, 256)
        self._batch_norm2 = nn.BatchNorm1d(256)

        # Third dense layer
        self._dense3 = nn.Linear(256, 64)
        self._batch_norm3 = nn.BatchNorm1d(64)

        # Output layer
        self._dense4 = nn.Linear(64, 1)

        # Activation functions and dropout
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(dropout)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, embedding_dim)

        # First dense block
        x = self._dense1(x)
        x = self._batch_norm1(x)
        x = self._relu(x)
        x = self._dropout(x)

        # Second dense block
        x = self._dense2(x)
        x = self._batch_norm2(x)
        x = self._relu(x)
        x = self._dropout(x)

        # Third dense block
        x = self._dense3(x)
        x = self._batch_norm3(x)
        x = self._relu(x)
        x = self._dropout(x)

        # Output layer
        x = self._dense4(x)
        x = self._sigmoid(x)

        return x.squeeze(-1)

    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > threshold).float()
            return predictions


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any
) -> List[float]:
    """
    Train the model and display informative progress bars for both training and validation.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs for training.
        criterion (nn.Module): Loss function to optimize.
        optimizer (Optimizer): Optimizer for model parameters.
        scheduler (Any): Learning rate scheduler.

    Returns:
        List[float]: Validation losses recorded after each epoch.
    """
    val_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training Loop
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc="Training", unit="batch") as train_bar:
            for batch_x, batch_y in train_bar:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Track the loss
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with tqdm(val_loader, desc="Validating", unit="batch") as val_bar:
            with torch.no_grad():
                for val_x, val_y in val_bar:
                    outputs = model(val_x)
                    loss = criterion(outputs, val_y)

                    # Track the loss
                    val_loss += loss.item()
                    val_bar.set_postfix(loss=loss.item())

        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Scheduler step
        scheduler.step(val_loss)

        # Epoch summary
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return val_losses

def main():
    # Define model, criterion, optimizer, scheduler
    model = ProModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Train the model
    val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Print validation losses
    print("Validation losses:", val_losses)