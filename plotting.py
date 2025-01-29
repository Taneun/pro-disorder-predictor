import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import torch
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def plot_roc_curve(model, test_loader):
    """
    Create and plot ROC curve using the best model.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader containing test data
    """
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch["embedding"])
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    # Create plot
    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='royalblue', width=2.5)
    ))

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # Customize layout
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        title_font=dict(size=24, color='darkslategray'),
        xaxis=dict(
            title='False Positive Rate',
            title_font=dict(size=18, color='darkslategray'),
            tickfont=dict(size=14, color='gray'),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='True Positive Rate',
            title_font=dict(size=18, color='darkslategray'),
            tickfont=dict(size=14, color='gray'),
            gridcolor='lightgray'
        ),
        legend=dict(
            font=dict(size=14),
            bordercolor='lightgray',
            borderwidth=1
        ),
        template='plotly_white'
    )

    # Print AUC score
    print(f"AUC Score: {roc_auc:.4f}")
    # Calculate balanced accuracy
    tpr = np.array(tpr)
    balanced_accuracy = (tpr + (1 - fpr)) / 2
    best_threshold = np.argmax(balanced_accuracy)
    print(f"Best Threshold: {best_threshold}")
    print(f"Balanced Accuracy: {balanced_accuracy[best_threshold]:.4f}")

    # fig.show()
    fig.write_html("mlp_roc_curve.html")
    fig.write_image("mlp_roc_curve.png", width=800, height=600)


def plot_losses(train_losses, val_losses):
    epochs = np.arange(1, len(train_losses) + 1)

    fig = go.Figure()

    # Add training loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='royalblue', width=2.5),
        marker=dict(size=8, symbol='circle', color='royalblue')
    ))

    # Add validation loss trace
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='firebrick', width=2.5, dash='dash'),
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
    # fig.show()
    fig.write_html("mlp_losses.html")
    fig.write_image("mlp_losses.png", width=800, height=600)


def preprocess_data_for_pca(data, pca_type="protein"):
    """
    Prepares data for PCA by processing embeddings and labels at either protein or amino acid level.

    Args:
        data (list of dict): Each dict contains:
            - "id" (str): Protein identifier
            - "rep" (torch.Tensor): Shape (n_amino_acids, embedding_dim)
            - "labels" (torch.Tensor): Shape (n_amino_acids, 1)
        pca_type (str): Type of PCA to perform:
            - "protein": Returns protein-level embeddings (mean over sequence)
            - "amino": Returns amino acid-level embeddings (all amino acids)

    Returns:
        tuple: (embeddings, labels)
            - If pca_type="protein":
                embeddings: np.array (n_proteins, embedding_dim)
                labels: np.array (n_proteins,)
            - If pca_type="amino":
                embeddings: np.array (total_amino_acids, embedding_dim)
                labels: np.array (total_amino_acids,)
    """
    if pca_type not in ["protein", "amino"]:
        raise ValueError("pca_type must be either 'protein' or 'amino'")

    if pca_type == "protein":
        # Process at protein level (mean over sequence)
        embeddings = []
        labels = []

        for protein_file in data:
            # Mean over sequence for embeddings
            embeddings.append(protein_file["rep"].mean(dim=0).numpy())
            # Mean over sequence for labels
            labels.append(np.mean(protein_file["labels"]))

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)

    else:  # amino level
        # Process at amino acid level (all individual AAs)
        embeddings = []
        labels = []

        for protein_file in data:
            # Add all amino acid embeddings
            embeddings.append(protein_file["rep"].numpy())
            # Add all amino acid labels
            labels.append(np.array(protein_file["labels"]).flatten())

        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)

    return embeddings, labels


def plot_embedding_pca(data_dir, num_components=2, pca_type="protein"):
    """
    Plots a PCA projection of high-dimensional embeddings using Plotly.
    Supports both protein-level and amino acid-level visualization.

    Args:
        data (list of dict): Each dict contains:
            - "id" (str): Protein identifier
            - "rep" (torch.Tensor): Shape (n_amino_acids, embedding_dim)
            - "labels" (torch.Tensor): Shape (n_amino_acids, 1)
        num_components (int): Number of PCA components to project onto (default=2).
        pca_type (str): Type of PCA to perform: "protein" or "amino" (default="protein").
    """
    protein_files = list(data_dir.glob("*.pt"))
    data = [torch.load(protein_file) for protein_file in protein_files]
    # Get preprocessed data
    embeddings, labels = preprocess_data_for_pca(data, pca_type=pca_type)

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(embeddings)

    # Initialize DataFrame with PCA results
    pca_df = pd.DataFrame({
        "PC1": pca_result[:, 0],
        "PC2": pca_result[:, 1],
    })

    # Add labels and protein information based on PCA type
    if pca_type == "protein":
        pca_df["% of Disorder"] = labels
        pca_df["Protein_ID"] = [item["id"] for item in data]
        color_column = "% of Disorder"
        hover_data = {"Protein_ID": True}
    else:
        pca_df["Disorder"] = labels
        # Create protein IDs and positions for amino acids
        protein_ids = []
        positions = []
        current_pos = 0
        for item in data:
            num_aas = len(item["rep"])
            protein_ids.extend([item["id"]] * num_aas)
            positions.extend(range(current_pos, current_pos + num_aas))
            current_pos += num_aas
        pca_df["Protein_ID"] = protein_ids
        pca_df["Position"] = positions
        color_column = "Disorder" if pca_type == "protein" else "Protein_ID"
        hover_data = {"Protein_ID": True, "Position": True}

    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_ * 100

    # Create Plotly scatter plot
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_column,
        color_continuous_scale=px.colors.sequential.Plasma_r,
        hover_data=hover_data,
        title=f"PCA of {'Protein' if pca_type == 'protein' else 'Amino Acid'} Embeddings",
        template="simple_white",
    )

    # Update marker size based on PCA type
    marker_size = 10 if pca_type == "protein" else 5

    # Update layout
    fig.update_traces(
        marker=dict(
            size=marker_size,
            opacity=0.8,
            line=dict(width=0)
        )
    )

    fig.update_layout(
        font=dict(family="Arial", size=14),
        title=dict(font=dict(size=18, color="black")),
        xaxis=dict(
            title=f"Principal Component 1 ({explained_var[0]:.1f}%)",
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
            showgrid=True,
        ),
        yaxis=dict(
            title=f"Principal Component 2 ({explained_var[1]:.1f}%)",
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
            showgrid=True,
        ),
        legend=dict(
            title=color_column,
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="white",
    )

    # Show plot
    # fig.show()
    fig.write_html(f"pca_{pca_type}.html")


def visualize_mlp_plotly(input_dim=1280, hidden_dims=[448, 224, 112], output_dim=1):
    """
    Visualize a Multi-Layer Perceptron architecture using Plotly.

    Args:
        input_dim (int): Number of input neurons
        hidden_dims (list): List of neurons in hidden layers
        output_dim (int): Number of output neurons
    """
    # List all layers
    all_layers = [input_dim] + hidden_dims + [output_dim]
    n_layers = len(all_layers)

    # Create node positions
    node_x = []
    node_y = []
    node_color = []

    # Colors for different layers
    colors = ['rgb(255,153,255)', 'rgb(153,187,255)', 'rgb(153,187,255)',
              'rgb(153,187,255)', 'rgb(187,255,187)']

    # Create nodes for each layer
    for i, n_neurons in enumerate(all_layers):
        x = i * 2  # Horizontal spacing between layers

        # Calculate vertical positions for neurons
        if n_neurons == 1:
            ys = [0]
        else:
            ys = np.linspace(-n_neurons / 2, n_neurons / 2, n_neurons)

        # Add nodes
        for j, y in enumerate(ys):
            node_x.append(x)
            node_y.append(y)
            node_color.append(colors[i])

    # Create edges (connections between layers)
    edge_x = []
    edge_y = []

    # Connect each layer to the next
    for i in range(n_layers - 1):
        layer_size = all_layers[i]
        next_layer_size = all_layers[i + 1]

        # Get positions for current and next layer
        current_ys = np.linspace(-layer_size / 2, layer_size / 2, layer_size)
        next_ys = np.linspace(-next_layer_size / 2, next_layer_size / 2, next_layer_size)

        # Connect each neuron to all neurons in next layer
        for y1 in current_ys:
            for y2 in next_ys:
                edge_x.extend([i * 2, (i + 1) * 2, None])
                edge_y.extend([y1, y2, None])

    # Create the figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='rgb(210,210,210)', width=1),
        hoverinfo='none'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=20,
            color=node_color,
            line=dict(color='rgb(50,50,50)', width=1)
        ),
    ))

    # Add layer labels
    annotations = []
    for i, n_neurons in enumerate(all_layers):
        label = f"Input Layer<br>({n_neurons})" if i == 0 else \
            f"Output Layer<br>({n_neurons})" if i == len(all_layers) - 1 else \
                f"Hidden Layer {i}<br>({n_neurons})<br>+ BatchNorm<br>+ ReLU<br>+ Dropout"

        annotations.append(dict(
            x=i * 2,
            y=max(node_y) + 2,
            xref="x",
            yref="y",
            text=label,
            showarrow=False,
            font=dict(size=12)
        ))

    # Update layout
    fig.update_layout(
        title="Multi-Layer Perceptron Architecture",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    return fig


# Create and show the visualization
# fig = visualize_mlp_plotly()
# fig.show()

# To save the visualization as HTML
# fig.write_html("mlp_architecture.html")
