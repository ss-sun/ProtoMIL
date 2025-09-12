import os
import argparse
import torch
from sparse_autoencoder import SparseAutoencoder

from data_dir_dict import data_dir_dict


def extract_concept_activation(src_img_embed_dir, dest_concept_embed_dir, sae_model, device):
    img_files = [file for file in os.listdir(src_img_embed_dir) if file.endswith('.pt')]
    for img_file in img_files:
        img_activation = torch.load(os.path.join(src_img_embed_dir, img_file))
        img_activation = img_activation.to(device)
        concept_embedding, reconstructions = sae_model(img_activation)
        concept_embedding = concept_embedding.squeeze().detach().cpu()
        dest_file = os.path.join(dest_concept_embed_dir, img_file)
        torch.save(concept_embedding, dest_file)



def load_sae_model(ckpt_path, autoencoder_input_dim, n_learned_features, device):
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=1).to(device)
    state_dict = torch.load(ckpt_path)
    autoencoder.load_state_dict(state_dict)
    return autoencoder


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--which_machine", type=str, default="workstation", choices=["workstation", "mlcloud", "laptop"])
    parser.add_argument("--sae_dataset", type=str, default='camelyon_panda')
    parser.add_argument("--autoencoder_input_dim", type=int, default=512, help="input dimension of the autoencoder, we use conch, input dimension is 512")
    parser.add_argument("--expansion_factor", type=int, default=4)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    data_dir = data_dir_dict["sae_dataset"][args.sae_dataset]
    print(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_img_embed_dir = data_dir["activation_dir"]
    dest_concept_embed_dir = data_dir["concept_embeddings"]
    ckpt_path = data_dir["sae_path"]
    print("ckpt_path", ckpt_path)

    autoencoder_input_dim = args.autoencoder_input_dim
    n_learned_features = autoencoder_input_dim * args.expansion_factor

    if not os.path.exists(dest_concept_embed_dir):
        os.makedirs(dest_concept_embed_dir)

    sae_model = load_sae_model(ckpt_path, autoencoder_input_dim, n_learned_features, device)
    extract_concept_activation(src_img_embed_dir, dest_concept_embed_dir, sae_model, device)
    print("Concept activations saved at: ", dest_concept_embed_dir)
    print(sae_model)

