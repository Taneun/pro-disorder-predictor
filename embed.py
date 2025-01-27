import torch
import esm
def generate_embedding():
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data_dic = {
        "protein1": ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",[0, 1]),
        "protein2": ("KALTARQQEVFDLIR",[0, 1]),
        "protein3": ("KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHIVSGASRGIRLLQEE",[0, 1]),
    }

    #convert data to list of tuples (id, sequence)
    data = [(k, v[0]) for k, v in data_dic.items()]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representation = token_representations[i, 1 : tokens_len - 1].mean(1)
        #replace the sequence representations in the data dictionary
        sequence_representations.append({"id": data[i][0], "rep": sequence_representation, "labels": data_dic[data[i][0]][1]})
