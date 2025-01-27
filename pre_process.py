import json
import numpy as np

"""
Usage API for pre_process.py

This module processes protein data from a JSON file, prepares it for modeling, and optionally generates a FASTA file.

Usage example:

from pre_process import preprocess_data

    # Process data and optionally create a FASTA file
    model_ready_data = preprocess_data(json_path, output_fasta_path)

    # If FASTA file creation is not needed, simply pass None
    model_ready_data = preprocess_data(json_path)

"""

def load_json_data(path: str) -> dict:
    """
    Loads a JSON file from the specified path.
    """
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def extract_disordered_regions(sample: dict) -> np.ndarray:
    """
    Extracts start and end indices of disordered regions for a single protein sample.
    """
    return np.array([(region['start'], region['end']) for region in sample.get('regions', [])])


def generate_labels_for_sequence(sequence: str, disordered_regions: np.ndarray) -> np.ndarray:
    """
    Creates a labeled sequence using NumPy.
    """
    labels = np.zeros(len(sequence), dtype=int)
    if disordered_regions.size > 0:
        for start, end in disordered_regions:
            labels[start - 1:end] = 1
    return labels


def extract_protein_data_from_json(json_file: dict) -> dict:
    """
    Extracts sequences and their labeled regions from the JSON file.
    """
    filtered_data = {}
    for sample in json_file.get('data', []):
        if 'alphafold_very_low_content' in sample:
            sequence = sample['sequence']
            disordered_regions = extract_disordered_regions(sample)
            labeled_sequence = generate_labels_for_sequence(sequence, disordered_regions)
            filtered_data[sample['acc']] = (sequence, labeled_sequence.tolist())
    return filtered_data


def write_fasta_file(data: dict, output_path: str):
    """
    Creates a FASTA file where each protein sequence is labeled with its `acc` as the header.
    """
    with open(output_path, 'w') as fasta_file:
        for acc, (sequence, labeled_sequence) in data.items():
            fasta_file.write(f">{acc}\n{sequence}\n{''.join(map(str, labeled_sequence))}\n")


def prepare_model_data(data: dict) -> dict:
    """
    Prepares data for modeling, returning sequences and their labels as a dictionary.
    """
    return {acc: {'sequence': sequence, 'labels': labels} for acc, (sequence, labels) in data.items()}


def preprocess_data(json_path: str, output_fasta_path: str = None) -> dict:
    """
    Processes the JSON data, prepares it for modeling, optionally creates a FASTA file, and returns the model-ready data.
    
    Args:
        json_path (str): The path to the input JSON file.
        output_fasta_path (str, optional): The path to the output FASTA file. If None, no FASTA file is created.
    
    Returns:
        dict: The model-ready data, with protein accession numbers as keys and sequences/labels as values.
    """
    # Step 1: Load the JSON data
    json_data = load_json_data(json_path)

    # Step 2: Extract sequences and labeled regions
    extracted_data = extract_protein_data_from_json(json_data)

    # Step 3: Prepare data for modeling
    model_ready_data = prepare_model_data(extracted_data)

    # Step 4: Optionally create a FASTA file
    if output_fasta_path:
        write_fasta_file(extracted_data, output_fasta_path)

    return model_ready_data


# if __name__ == "__main__":
    
#     json_path = "C:/Users/danas/OneDrive/Desktop/pro-disorder-predictor/DisProt_release_2024_12_Consensus_without_includes.json"
#   # Update with your JSON file path
#     output_fasta = "C:/Users/danas/OneDrive/Desktop/pro-disorder-predictor/output.fasta"

#     # Step 1: Load the JSON file
#     json_data = load_json_file(json_path)

#     # Step 2: Extract sequences and labeled regions
#     extracted_data = extract_data_from_json(json_data)

#     # Step 3: Create a FASTA file
#     write_fasta_file(extracted_data, output_fasta)

#     # Step 4: Prepare data for modeling
#     model_ready_data = prepare_model_data(extracted_data)
#     print(model_ready_data['P49913'])