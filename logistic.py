import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report


def load_and_prepare_data(pt_file):
    """
    Load data from PT file and prepare it for training.
    Returns features, labels, and protein IDs arrays.
    """
    # Load the saved embeddings
    sequence_representations = torch.load(pt_file)

    # Initialize lists to store features, labels, and protein IDs
    all_features = []
    all_labels = []
    protein_ids = []

    # Process each protein sequence
    for seq in sequence_representations:
        # Convert embeddings to numpy array
        features = seq["rep"].numpy()
        labels = seq["labels"]

        # Append features and labels for each amino acid
        all_features.append(features)
        all_labels.append(labels)

        # Repeat protein ID for each amino acid in the sequence
        protein_ids.extend([seq["id"]] * len(labels))

    # Concatenate all features and labels
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    return X, y, np.array(protein_ids)


def train_and_evaluate(X, y, protein_ids, n_splits=5):
    """
    Train and evaluate logistic regression model using group k-fold cross validation.
    Ensures amino acids from the same protein stay together.
    """
    # Initialize GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)

    # Store results for each fold
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, protein_ids)):
        # Split data while keeping proteins together
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize and train logistic regression model
        model = LogisticRegression(max_iter=1000, multi_class='ovr')
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Generate classification report
        report = classification_report(y_test, y_pred)

        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'report': report,
            'model': model
        })

        print(f"\nFold {fold + 1} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

    return fold_results


def main(pt_file_path):
    """
    Main function to run the entire pipeline.
    """
    # Load and prepare data
    print("Loading data...")
    X, y, protein_ids = load_and_prepare_data(pt_file_path)

    print(f"Loaded {len(protein_ids)} amino acids from {len(np.unique(protein_ids))} proteins")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of unique labels: {len(np.unique(y))}")

    # Train and evaluate model
    print("\nTraining and evaluating model...")
    results = train_and_evaluate(X, y, protein_ids)

    # Calculate and print average accuracy across folds
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\nAverage accuracy across folds: {avg_accuracy:.4f}")

    return results


if __name__ == "__main__":
    pt_file_path = "embeddings.pt"
    results = main(pt_file_path)