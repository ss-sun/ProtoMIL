import os
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch


class WSIEmbeddingDataset(Dataset):
    def __init__(self,
                 df,
                 data_source,
                 index_col='slide_id',
                 target_col='label',
                 perturbed_neuron = []
                 ):
        """
        Args:
        """
        self.data = df
        self.data_source = data_source
        self.index_col = index_col
        self.target_col = target_col
        self.ext = '.pt'  # use pt format
        self.perturbed_neuron = perturbed_neuron


    def __len__(self):
        return len(self.data)

    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        if self.target_col is None:
            return []
        return self.data.loc[ids, self.target_col]

    def get_feat_path(self, slide_id):
        feat_path = os.path.join(self.data_source, slide_id + self.ext)
        return feat_path

    def __getitem__(self, idx):
        slide_id = self.get_ids(idx)
        label = self.get_labels(idx)
        slide_ids = [slide_id]

        all_features = []

        for slide_id in slide_ids:
            feat_path = self.get_feat_path(slide_id)
            features = torch.load(feat_path, weights_only=True)
            if len(self.perturbed_neuron) > 0:
                features[:, self.perturbed_neuron] = 0
            if features.device.type == 'cuda':
                features = features.cpu().detach().numpy()
            all_features.append(features)

        all_features = torch.from_numpy(np.concatenate(all_features, axis=0))

        if len(all_features.size()) == 3:
            all_features = all_features.squeeze(0)

        out = {'feature': all_features, 'label': label}

        return out


if __name__ == '__main__':
    import pandas as pd
    from torch.utils.data import DataLoader

    # train_val_data_source = "/home/susu/datasets/camelyon16_clam_level1_256/features/conch_v1/pt_files"
    # test_data_source = "/home/susu/datasets/camelyon16_clam_level1_256/features/conch_v1/pt_files"

    data_source = "/home/susu/datasets/camelyon16_clam_level1_256/features/conch_v1/sae_concept_embeddings"

    # data_source = "/home/susu/datasets/camelyon16_clam_level1_256/features/conch_v1/pt_files"

    train_csv_path = "/home/susu/datasets/camelyon16_clam_level1_256/split_csv/train.csv"  # for train.py si-mil
    valid_csv_path = "/home/susu/datasets/camelyon16_clam_level1_256/split_csv/valid.csv"  # for train.py si-mil
    test_gt_path =  "/home/susu/datasets/camelyon16_clam_level1_256/split_csv/test.csv"  # for train.py si-mil

    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)
    test_df = pd.read_csv(test_gt_path)

    print(train_df.head())

    perturbed_neuron = [267, 1070, 1139, 1238]

    train_dataset = WSIEmbeddingDataset(data_source=data_source,
                                        df=train_df,
                                        index_col="slide_id",
                                        target_col="label",
                                        perturbed_neuron=perturbed_neuron)

    valid_dataset = WSIEmbeddingDataset(data_source=data_source,
                                        df=valid_df,
                                        index_col="slide_id",
                                        target_col="label",
                                        perturbed_neuron=perturbed_neuron)

    test_dataset = WSIEmbeddingDataset(data_source=data_source,
                                       df=test_df,
                                       index_col="slide_id",
                                       target_col="label",
                                       perturbed_neuron=perturbed_neuron)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    print(len(train_dataloader))
    print(len(valid_dataloader))
    print(len(test_dataloader))
    for i, data in enumerate(train_dataloader):
        # print(data)
        print(type(data['feature']))
        print(data["feature"].shape)
        print(data["label"].shape)
        print(data["label"])
    #     # print(data["concept_features"][0, 0, :])
        if i == 5:
            break

