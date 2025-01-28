from typing import Tuple, List

from tqdm import tqdm
from typing import *
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.graph_objects as go



class AminoAcidDataset(Dataset):
    def __init__(self, data_list):
        """
        Initialize the dataset from a list of protein dictionaries.

        Args:
            data_list: List of dictionaries containing protein data
                      Each dict has keys: protein_id, embedding, label
        """
        self.samples = []

        # Process each protein in the data list
        for protein_dict in data_list:
            protein_id = protein_dict["id"]
            embeddings = protein_dict["rep"]  # Shape: (protein_length, 1280)
            labels = protein_dict["labels"]  # Shape: (protein_length, 1)

            # Process each amino acid in the protein
            for aa_idx in range(len(embeddings)):
                self.samples.append({
                    "protein_id": protein_id,
                    "aa_index": aa_idx,
                    "embedding": embeddings[aa_idx],  # Shape: (1280,)
                    "label": labels[aa_idx]  # Convert to scalar
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "protein_id": sample["protein_id"],
            "aa_index": sample["aa_index"],
            "embedding": sample["embedding"],
            "label": sample["label"]
        }


def create_stratified_dataloaders(data_list, batch_size=32, train_ratio=0.7, val_ratio=0.15,
                                  test_ratio=0.15, random_state=42):
    """
    Create stratified train/validation/test DataLoaders.

    Args:
        data_list: List of protein dictionaries
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = AminoAcidDataset(data_list)

    # Prepare data for stratification
    all_labels = [sample["label"] for sample in full_dataset.samples]
    all_indices = np.arange(len(full_dataset))

    # First split: train and temp (val + test)
    train_indices, temp_indices = train_test_split(
        all_indices,
        train_size=train_ratio,
        stratify=[all_labels[i] for i in all_indices],
        random_state=random_state
    )

    # Second split: val and test from temp
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_adjusted,
        stratify=[all_labels[i] for i in temp_indices],
        random_state=random_state
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


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
) -> tuple[list[float], list[float]]:
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
    train_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training Loop
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc="Training", unit="batch", colour='blue') as train_bar:
            for batch in train_bar:
                optimizer.zero_grad()

                # Extract features and labels from batch dictionary
                batch_x = batch["embedding"]
                batch_y = batch["label"]

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs.float(), batch_y.float())

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Track the loss
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with tqdm(val_loader, desc="Validating", unit="batch", colour='green') as val_bar:
            with torch.no_grad():
                for batch in val_bar:
                    # Extract features and labels from batch dictionary
                    val_x = batch["embedding"]
                    val_y = batch["label"]

                    outputs = model(val_x)
                    loss = criterion(outputs.float(), val_y.float())

                    # Track the loss
                    val_loss += loss.item()
                    val_bar.set_postfix(loss=loss.item())

        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        # Scheduler step
        scheduler.step(val_loss)

        # Epoch summary
        print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return val_losses, train_losses


def plot_losses(train_losses, val_losses):
    epochs = np.arange(1, len(train_losses) + 1)

    fig = go.Figure()

    # Add training loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8, symbol='circle', color='royalblue')
    ))

    # Add validation loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='firebrick', width=2, dash='dash'),
        marker=dict(size=8, symbol='square', color='firebrick')
    ))

    # Customize layout for beauty
    fig.update_layout(
        title='Training and Validation Loss Over Epochs',
        title_font=dict(size=24, color='darkslategray'),
        xaxis=dict(
            title='Epochs',
            title_font=dict(size=18, color='darkslategray'),
            tickfont=dict(size=14, color='gray'),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Loss',
            title_font=dict(size=18, color='darkslategray'),
            tickfont=dict(size=14, color='gray'),
            gridcolor='lightgray'
        ),
        legend=dict(
            font=dict(size=14),
            bordercolor='lightgray',
            borderwidth=1
        ),
        template='plotly_white',
        hovermode='x unified'
    )

    # Add annotations for the lowest validation loss
    min_val_loss = min(val_losses)
    min_epoch = epochs[np.argmin(val_losses)]
    fig.add_annotation(
        x=min_epoch,
        y=min_val_loss,
        text=f"Min Val Loss: {min_val_loss:.4f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor='firebrick',
        font=dict(size=12, color='firebrick'),
        ax=20,
        ay=-30
    )

    # Show the figure
    fig.show()







def main():
    data = torch.load('embeddings_cpu.pt')

    # Split data into train, validation, and test sets
    train_loader, val_loader, test_loader = create_stratified_dataloaders(data)

    # Define model, criterion, optimizer, scheduler
    model = ProModel()
    criterion = nn.BCELoss()  # Changed from BCEWithLogitsLoss since we already have sigmoid
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Train the model
    val_losses, train_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)




if __name__ == "__main__":
    main()