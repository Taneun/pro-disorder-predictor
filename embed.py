import torch
import esm
from pre_process import preprocess_data
from tqdm import tqdm
import argparse

def embed_data(input_file: str):
    # Load the ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disables dropout for deterministic results
    sequence_representations = []

    # Preprocess the data
    print(f"Preprocessing data from {input_file}...")
    data_dic = preprocess_data(input_file)

    # Convert data to list of tuples (id, sequence)
    data = [(k, v[0]) for k, v in data_dic.items()]
    chunks = list(chunk_list(data, 1))
    for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
        prot_id = chunk[0][0]
        batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        print("Extracting embeddings...")
        # token_representations = []
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        print("Generating per-sequence representations...")

        # Generate per-sequence representations via averaging
        sequence_representation = token_representations[0, 1:batch_lens[0] - 1]

        # Create single protein dictionary
        protein_data = {
            "id": prot_id,
            "rep": sequence_representation.cpu(),
            "labels": data_dic[prot_id][1]
        }

        # Save the embeddings
        output_file = f"post_embedding/{prot_id}.pt"
        print(f"Saving embeddings to {output_file}...")
        torch.save(protein_data, output_file)
        print("Embeddings saved successfully!")


def chunk_list(list, chunk_size):
    for i in range(0, len(list), chunk_size):
        yield list[i:i + chunk_size]


if __name__ == "__main__":
    # Command-line argument parsing
    # parser = argparse.ArgumentParser(description="Generate protein embeddings using ESM2.")
    # parser.add_argument(
    #     "--input_file", type=str, required=True,
    #     help="Path to the input JSON file with protein data."
    # )
    #
    # args = parser.parse_args()
    # embed_data(input_file=args.input_file)
    embed_data("DisProt_release_2024_12_Consensus_without_includes.json")