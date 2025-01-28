from typing import Tuple, List
from embed import *
from plotting import *
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
import optuna
from copy import deepcopy


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


class ModelCheckpoint:
    def __init__(self, filepath):
        self.filepath = filepath
        self.best_val_loss = float('inf')
        self.best_model = None

    def update(self, model, val_loss):
        """Update the best model if current validation loss is better."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': self.best_model,
                'val_loss': self.best_val_loss
            }, self.filepath)
            return True
        return False

    def load_best_model(self, model):
        """Load the best model weights into the provided model."""
        checkpoint = torch.load(self.filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

def create_stratified_dataloaders(data_list, batch_size=32, train_ratio=0.65, val_ratio=0.15,
                                  test_ratio=0.2, random_state=42):
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


# Updated ProModel class with configurable layer sizes
class ProModel(nn.Module):
    def __init__(self, embedding_dim=1280, dropout=0.2, hidden_dims=[512, 256, 64]):
        super(ProModel, self).__init__()

        layers = []
        current_dim = embedding_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # Create layer sequence
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x.squeeze(-1)


# Updated training function to use model checkpoint
def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        checkpoint: ModelCheckpoint = None
) -> Tuple[List[float], List[float]]:
    val_losses = []
    train_losses = []

    epoch_bar = tqdm(total=epochs, desc="Epoches", unit="epoch", colour='blue', position=0)

    for epoch in range(epochs):
        # Training Loop
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            batch_x = batch["embedding"]
            batch_y = batch["label"]

            outputs = model(batch_x)
            loss = criterion(outputs.float(), batch_y.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                val_x = batch["embedding"]
                val_y = batch["label"]

                outputs = model(val_x)
                loss = criterion(outputs.float(), val_y.float())

                val_loss += loss.item()

        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        # Update model checkpoint
        if checkpoint is not None:
            is_best = checkpoint.update(model, val_loss)
            if is_best:
                print(f"\nNew best model saved! (val_loss: {val_loss:.4f})")

        # Scheduler step
        scheduler.step(val_loss)
        epoch_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}'
        })

        # Update epoch progress
        epoch_bar.update()

    epoch_bar.close()

    return val_losses, train_losses


def objective(trial, data, epochs=10, batch_size=32):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        data: Training data
        epochs: Number of epochs
        batch_size: Batch size
    """
    # Define hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    hidden_dims = [
        trial.suggest_int('hidden_dim1', 128, 512, step=64),
        trial.suggest_int('hidden_dim2', 64, 256, step=32),
        trial.suggest_int('hidden_dim3', 32, 128, step=16),
    ]

    # Create model and training components
    model = ProModel(embedding_dim=1280, dropout=dropout, hidden_dims=hidden_dims)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Create dataloaders
    train_loader, val_loader, _ = create_stratified_dataloaders(
        data, batch_size=batch_size
    )

    # Train model
    val_losses, _ = train_model(
        model, train_loader, val_loader, epochs,
        criterion, optimizer, scheduler
    )

    return min(val_losses)  # Return best validation loss


def tune_hyperparameters(data, n_trials=100):
    """
    Tune hyperparameters using Optuna.

    Args:
        data: Training data
        n_trials: Number of optimization trials
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data), n_trials=n_trials)

    print("\n*** Hyperparameter Tuning Results ***")
    print("\tBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


def main():
    # embed_data('DisProt_release_2024_12_Consensus_without_includes.json', 'embeddings_overnight.pt')
    data = torch.load('embeddings_overnight.pt')
    # cpu_data = torch.load('embeddings_cpu.pt')

    # Split data into train, validation, and test sets
    train_loader, val_loader, test_loader = create_stratified_dataloaders(data)

    best_params = tune_hyperparameters(data, n_trials=100)

    # Create model with best parameters
    model = ProModel(
        embedding_dim=1280,
        dropout=best_params['dropout'],
        hidden_dims=[
            best_params['hidden_dim1'],
            best_params['hidden_dim2'],
            best_params['hidden_dim3']
        ]
    )

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.BCELoss()  # Changed from BCEWithLogitsLoss since we already have sigmoid
    #
    checkpoint = ModelCheckpoint('best_model.pt')

    # Update training call
    val_losses, train_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint=checkpoint  # Add this line
    )

    # Load best model for evaluation
    best_model = checkpoint.load_best_model(model)

    plot_roc_curve(best_model, test_loader)

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()