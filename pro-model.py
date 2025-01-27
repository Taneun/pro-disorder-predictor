import torch
import torch.nn as nn

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


def train_step(model, batch_x, batch_y, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    model = ProModel()

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_x = torch.FloatTensor(your_embeddings)  # shape: (batch_size, 1280)
    batch_y = torch.FloatTensor(your_labels)  # shape: (batch_size,)

    loss = train_step(model, batch_x, batch_y, criterion, optimizer)