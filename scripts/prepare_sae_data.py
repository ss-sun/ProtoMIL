import os
import h5py
import torch

def convert_h5_to_pt(src_h5_features):
    dest_folder = os.path.join(os.path.dirname(src_h5_features), "features_conch_v1_pt")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)

    h5_files = [f for f in os.listdir(src_h5_features) if f.endswith('.h5')]
    print(f"Found {len(h5_files)} .h5 files in {src_h5_features}")

    for i in range(len(h5_files)):
        h5_file = h5_files[i]
        file_name = os.path.splitext(h5_file)[0]
        h5_path = os.path.join(src_h5_features, h5_file)

        with h5py.File(h5_path, 'r') as f:
            feat = f['features'][:]
            # need to normalize the features to L2 norm 1.0, important for training SAE
            feat = feat / torch.norm(torch.tensor(feat), dim=-1, keepdim=True)
            feat = torch.tensor(feat, dtype=torch.float32)
            dest_file_path = os.path.join(dest_folder, f"{file_name}.pt")
            torch.save(feat, dest_file_path)

if __name__ == "__main__":
    src_folders = [
        "/home/susu/datasets/trident_out_camelyon16/20x_256px_0px_overlap/features_conch_v1",
        "/home/susu/datasets/trident_out_panda/20x_256px_0px_overlap/features_conch_v1"
    ]
    for src_h5_features in src_folders:
        print(f"Converting features from {src_h5_features} to .pt files...")
        convert_h5_to_pt(src_h5_features)




