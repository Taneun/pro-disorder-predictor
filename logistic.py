import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, balanced_accuracy_score
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm


def load_and_prepare_data(directory):
    """
    Load data from all PT files in the specified directory.
    Returns features, labels, and protein IDs arrays.
    """
    # Initialize lists to store features, labels, and protein IDs
    all_features = []
    all_labels = []
    protein_ids = []

    # Get all PT files in directory
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]

    # Process each PT file
    for pt_file in tqdm(pt_files, desc="Loading protein data"):
        # Load the saved embeddings
        file_path = os.path.join(directory, pt_file)
        protein_data = torch.load(file_path)

        # Extract data
        features = protein_data["rep"]#.numpy()
        labels = protein_data["labels"]
        prot_id = protein_data["id"]

        # Append features and labels
        all_features.append(features)
        all_labels.append(labels)

        # Repeat protein ID for each amino acid in the sequence
        protein_ids.extend([prot_id] * len(labels))

    # Concatenate all features and labels
    # X = np.vstack(all_features)
    # y = np.concatenate(all_labels)
    X = np.vstack([t.cpu().numpy() for t in all_features])
    y = np.concatenate(all_labels)

    return X, y, np.array(protein_ids)


def plot_roc_curves(y_test, y_pred_proba, classes):
    """
    Plot ROC curves for each class.
    """
    plt.figure(figsize=(10, 8))

    # For binary classification
    if len(classes) == 2:
        # For binary classification, we only need the probability of the positive class
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2,
                 label=f'ROC curve (AUC = {roc_auc:0.2f})')

    # For multi-class classification
    else:
        # Binarize the labels for multi-class ROC
        y_test_bin = np.eye(len(classes))[np.searchsorted(classes, y_test)]

        # Calculate ROC curve and ROC area for each class
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f'ROC curve (class {class_label}, AUC = {roc_auc:0.2f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    return plt.gcf()


def train_and_evaluate_single_split(X, y, protein_ids, test_size=0.2):
    """
    Train and evaluate model using a single train-test split.
    Ensures amino acids from the same protein stay together.
    """
    # Get unique proteins
    unique_proteins = np.unique(protein_ids)

    # Split proteins into train and test sets
    train_proteins, test_proteins = train_test_split(unique_proteins, test_size=test_size, random_state=42)

    # Create masks for train and test sets
    train_mask = np.isin(protein_ids, train_proteins)
    test_mask = np.isin(protein_ids, test_proteins)

    # Split the data
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Get unique classes
    classes = np.unique(y)
    print(f"Unique classes: {classes}")

    # Initialize and train logistic regression model
    model = LogisticRegression(max_iter=1000, multi_class='ovr')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    # Generate ROC curve plot
    roc_plot = plot_roc_curves(y_test, y_pred_proba, classes)

    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("Classification Report:")
    print(report)

    # Save the plot
    plt.savefig('log_roc_curve.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return {
        'accuracy': accuracy,
        'report': report,
        'model': model,
        'roc_plot': roc_plot,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train logistic regression model on protein embeddings.')
    parser.add_argument('embedding_dir', type=str, help='Directory containing PT files with embeddings')
    args = parser.parse_args()

    # Load and prepare data
    print("Loading data from directory:", args.embedding_dir)
    X, y, protein_ids = load_and_prepare_data(args.embedding_dir)

    print(f"Loaded {len(protein_ids)} amino acids from {len(np.unique(protein_ids))} proteins")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of unique labels: {len(np.unique(y))}")

    # Train and evaluate model
    print("\nTraining and evaluating model...")
    results = train_and_evaluate_single_split(X, y, protein_ids)

    return results


if __name__ == "__main__":
    results = main()