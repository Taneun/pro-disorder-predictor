import json
import numpy as np


def load_json_file(path: str) -> dict:
    """
    Loads a JSON file from the specified path.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file as a Python dictionary.
    """
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_disordered_regions_ind(sample: dict) -> np.ndarray:
    """
    Extracts start and end indices of disordered regions for a single protein sample.

    Args:
        sample (dict): A dictionary representing a protein sample, which includes
                        the 'regions' key, containing the disordered regions.

    Returns:
        np.ndarray: A NumPy array of tuples, where each tuple represents the start 
                    and end indices of a disordered region.
    """
    return np.array([(region['start'], region['end']) for region in sample.get('regions', [])])


def label_sequence(sequence: str, disordered_regions: np.ndarray) -> np.ndarray:
    """
    Creates a labeled sequence using NumPy.

    Args:
        sequence (str): The amino acid sequence as a string.
        disordered_regions (np.ndarray): A NumPy array of tuples representing the
                                          start and end indices of disordered regions.

    Returns:
        np.ndarray: A NumPy array of labels (1 or 0) corresponding to each amino acid
                    in the sequence, where 1 indicates the amino acid is in a disordered
                    region and 0 otherwise.
    """
    labels = np.zeros(len(sequence), dtype=int)
    if disordered_regions.size > 0:
        # Convert 1-based to 0-based indexing and label disordered regions
        for start, end in disordered_regions:
            labels[start - 1:end] = 1
    return labels


def extract_data_from_json(json_file: dict) -> dict:
    """
    Extracts sequences and their labeled regions from the JSON file.

    Args:
        json_file (dict): The JSON data loaded as a Python dictionary.

    Returns:
        dict: A dictionary where the keys are protein accession numbers (`acc`) and
              the values are tuples containing the sequence and the corresponding labeled sequence.
    """
    filtered_data = {}
    for sample in json_file.get('data', []):
        if 'alphafold_very_low_content' in sample:
            sequence = sample['sequence']
            disordered_regions = get_disordered_regions_ind(sample)
            labeled_sequence = label_sequence(sequence, disordered_regions)
            filtered_data[sample['acc']] = (sequence, labeled_sequence.tolist())
    return filtered_data


def create_fasta_file(data: dict, output_path: str):
    """
    Creates a FASTA file where each protein sequence is labeled with its `acc` as the header.

    Args:
        data (dict): A dictionary where the keys are protein accession numbers (`acc`) and
                     the values are tuples containing the sequence and labeled sequence.
        output_path (str): The file path where the FASTA file will be saved.
    """
    with open(output_path, 'w') as fasta_file:
        for acc, (sequence, labeled_sequence) in data.items():
            # Write the sequence and its corresponding labeled sequence to the FASTA file
            fasta_file.write(f">{acc}\n{sequence}\n{''.join(map(str, labeled_sequence))}\n")


def get_data_for_model(data: dict) -> dict:
    """
    Prepares data for modeling, returning sequences and their labels as a dictionary.

    Args:
        data (dict): A dictionary where the keys are protein accession numbers (`acc`) and
                     the values are tuples containing the sequence and labeled sequence.

    Returns:
        dict: A dictionary where the keys are protein accession numbers (`acc`) and the values
              are dictionaries containing the sequence and labels.
    """
    return {acc: {'sequence': sequence, 'labels': labels} for acc, (sequence, labels) in data.items()}


if __name__ == "__main__":
    
    json_path = "C:/Users/danas/OneDrive/Desktop/pro-disorder-predictor/DisProt_release_2024_12_Consensus_without_includes.json"
  # Update with your JSON file path
    output_fasta = "C:/Users/danas/OneDrive/Desktop/pro-disorder-predictor/output.fasta"

    # Step 1: Load the JSON file
    json_data = load_json_file(json_path)

    # Step 2: Extract sequences and labeled regions
    extracted_data = extract_data_from_json(json_data)

    # Step 3: Create a FASTA file
    create_fasta_file(extracted_data, output_fasta)

    # Step 4: Prepare data for modeling
    model_ready_data = get_data_for_model(extracted_data)
    print(len(model_ready_data))
