import torch
import esm
from pre_process import preprocess_data
from tqdm import tqdm
import gc
import argparse

# def embed_data(input_file: str):
#     # Load the ESM model
#     torch.hub.set_dir("../cache/")
#     model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#     batch_converter = alphabet.get_batch_converter()
#     model.cuda()
#     model.eval()  # Disables dropout for deterministic results
#
#     # Preprocess the data
#     print(f"Preprocessing data from {input_file}...")
#     data_dic = preprocess_data(input_file)
#
#     # Convert data to list of tuples (id, sequence)
#     data = [(k, v[0]) for k, v in data_dic.items()]
#     chunks = list(chunk_list(data, 1))
#     for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
#         prot_id = chunk[0][0]
#         batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
#         batch_tokens = batch_tokens.cuda()
#         batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#
#         # Extract per-residue representations (on CPU)
#         print("Extracting embeddings...")
#         # token_representations = []
#         with torch.no_grad():
#             results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#             token_representations = results["representations"][33]
#
#         # Generate per-sequence representations via averaging
#         print("\nGenerating per-sequence representations...")
#
#         # Generate per-sequence representations via averaging
#         sequence_representation = token_representations[0, 1:batch_lens[0] - 1]
#
#         # Create single protein dictionary
#         protein_data = {
#             "id": prot_id,
#             "rep": sequence_representation,
#             "labels": data_dic[prot_id][1]
#         }
#
#         # Save the embeddings
#         output_file = f"post_embedding/{prot_id}.pt"
#         print(f"Saving embeddings to {output_file}...")
#         torch.save(protein_data, output_file)
#         print("Embeddings saved successfully!")
#         # Clear some memory
#         del results, token_representations, sequence_representation, protein_data
#         torch.cuda.empty_cache()
#
#
def chunk_list(list, chunk_size):
    for i in range(0, len(list), chunk_size):
        yield list[i:i + chunk_size]

def embed_data(input_file: str):
    # Load the ESM model
    torch.hub.set_dir("../cache/")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    # Move model to GPU only when needed
    model.eval()  # Disables dropout for deterministic results

    # Enable memory efficient options
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        # Set memory allocator configuration
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available memory
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Preprocess the data
    print(f"Preprocessing data from {input_file}...")
    data_dic = preprocess_data(input_file)

    # Convert data to list of tuples (id, sequence)
    data = [(k, v[0]) for k, v in data_dic.items()]
    chunks = list(chunk_list(data, 1))

    for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
        try:
            prot_id = chunk[0][0]
            batch_labels, batch_strs, batch_tokens = batch_converter(chunk)

            # Process in smaller sub-batches if sequence is too long
            max_length = 1022  # Maximum sequence length to process at once
            seq_length = batch_tokens.size(1)

            if seq_length > max_length:
                # Process long sequences in segments
                representations = []
                for start in range(0, seq_length, max_length):
                    end = min(start + max_length, seq_length)
                    batch_segment = batch_tokens[:, start:end]

                    # Move segment to GPU
                    model.cuda()
                    batch_segment = batch_segment.cuda()

                    with torch.no_grad():
                        results = model(batch_segment, repr_layers=[33], return_contacts=True)
                        segment_representations = results["representations"][33]
                        representations.append(segment_representations.cpu())

                    # Clear GPU memory
                    del results, segment_representations
                    model.cpu()
                    torch.cuda.empty_cache()

                # Concatenate segments
                token_representations = torch.cat(representations, dim=1)
                sequence_representation = token_representations[0, 1:-1]

            else:
                # Process normally for shorter sequences
                model.cuda()
                batch_tokens = batch_tokens.cuda()

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]
                    sequence_representation = token_representations[0, 1:batch_lens[0] - 1].cpu()

                # Clear GPU memory
                del results, token_representations
                model.cpu()
                torch.cuda.empty_cache()

            # Create single protein dictionary
            protein_data = {
                "id": prot_id,
                "rep": sequence_representation,
                "labels": data_dic[prot_id][1]
            }

            # Save the embeddings
            output_file = f"post_embedding/{prot_id}.pt"
            print(f"Saving embeddings to {output_file}...")
            torch.save(protein_data, output_file)
            print("Embeddings saved successfully!")

            # Clear CPU memory
            del sequence_representation, protein_data
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: Out of memory for protein {prot_id}. Clearing cache and skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e


# if __name__ == "__main__":
    # Command-line argument parsing
    # parser = argparse.ArgumentParser(description="Generate protein embeddings using ESM2.")
    # parser.add_argument(
    #     "--input_file", type=str, required=True,
    #     help="Path to the input JSON file with protein data."
    # )
    #
    # args = parser.parse_args()
    # embed_data(input_file=args.input_file)
    # embed_data("DisProt_release_2024_12_Consensus_without_includes.json")