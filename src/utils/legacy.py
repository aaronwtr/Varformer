import torch.nn.functional as F

def apply_convert_to_dense_on_dict(dict_obj, shape):
    for key, value in dict_obj.items():
        dict_obj[key] = _convert_to_dense(value, shape)
    return dict_obj


def one_hot_encode(cat_list: list):
    vocab = {category: idx for idx, category in enumerate(cat_list)}
    indices = [vocab[category] for category in cat_list]
    num_classes = len(cat_list)
    one_hot_tensor = F.one_hot(torch.tensor(indices), num_classes=num_classes).float()
    return one_hot_tensor


def chunk_open_file(file: str):
    chunk_size = int(1e06)
    with open(file, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            chunk = chunk.splitlines()
            if not chunk:
                break


def get_protein_length_up(up):
    uniprot_api_url = f"https://www.uniprot.org/uniprot/{up}.fasta"

    response = requests.get(uniprot_api_url)
    try:
        if response.status_code == 200:
            output = response.text
            protein_sequence = output.split("\n")[1:]
            protein_sequence = "".join(protein_sequence)
            protein_length = len(protein_sequence)
            if protein_length == 0:
                raise ValueError(f"Protein length for {up} not found in the response.")

            return protein_length
        else:
            raise KeyError(f"Failed to retrieve protein sequence from UniProt. Status code: {response.status_code}")

    except ValueError:
        print(f"Protein length for {up} not found in the response.")
        return 0
    except KeyError:
        print(f"Failed to retrieve protein sequence from UniProt. Status code: {response.status_code}")
        return 0
