import pandas as pd
import os
import sys
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import torch.nn.functional as F
from sklearn.manifold import TSNE
from conch.open_clip_custom import tokenize, get_tokenizer, create_model_from_pretrained
import seaborn as sns
import shutil

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import h5py


os.environ['PYTHONPATH'] = "/opt/ASAP/bin"
# Optionally, add the path to sys.path
sys.path.append("/opt/ASAP/bin")
import multiresolutionimageinterface as mir



## 0: background, 1: normal, 2: tumor for Camelyon 16
## 0: background, 1: normal, 2: tumor for samples from Karolinska in PANDA
## 0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5 for samples from Radboud in PANDA




def get_basic_info(dataset_name, wsi_id, wsi_dir, gt_dir, wsi_ext):
    ### adapted from https://github.com/DIAGNijmegen/pathology-common/blob/master/examples/getting_started/1_asap.ipynb

    print("printing basic info for dataset: ", dataset_name)
    print("on wsi_id: ", wsi_id)

    reader = mir.MultiResolutionImageReader()

    wsi_path = os.path.join(wsi_dir, wsi_id + wsi_ext)
    wsi_img = reader.open(wsi_path)
    print('Image dimensions at level 0: {}'.format(wsi_img.getDimensions()))
    print('Image pixel spacing at level 0: {}'.format(wsi_img.getSpacing()))
    print('Numer of levels in the image: {} \n'.format(wsi_img.getNumberOfLevels()))

    gt_mask_file_path = os.path.join(gt_dir, wsi_id + "_mask" + wsi_ext)
    mask_img = reader.open(gt_mask_file_path)

    print('Min and max values found in the image are: {} and {}'.format(
        wsi_img.getMinValue(), wsi_img.getMaxValue()))
    print('Min and max values found in the mask are: {} and {} \n'.format(
        mask_img.getMinValue(),
        mask_img.getMaxValue()))  # for camelyon: three labels: background (0), healthy tissue (1), tumor (2)
    print('Amount of channels present in the image: {}'.format(
        wsi_img.getSamplesPerPixel()))
    print('Amount of channels present in the mask: {} \n'.format(
        mask_img.getSamplesPerPixel()))

    patch_level = 0
    img = wsi_img.getUCharPatch(0, 0,
                                     *wsi_img.getLevelDimensions(patch_level),
                                     patch_level).squeeze()
    print("img.shape: ", img.shape)  ## ds_mask.shape:  (12288, 3584, 3)

    extracted_patch = wsi_img.getUCharPatch(1000, 2000, 256, 256, patch_level)
    ### show the extracted patch
    print(type(extracted_patch))
    plt.imshow(extracted_patch)
    plt.title("extracted patch")
    plt.axis('off')
    plt.show()

    patch_level = 1
    img = wsi_img.getUCharPatch(0, 0, *wsi_img.getLevelDimensions(patch_level),patch_level).squeeze()
    print("img.shape: ", img.shape)

    extracted_patch = wsi_img.getUCharPatch(1000, 2000, 256, 256, patch_level)
    plt.imshow(extracted_patch)
    plt.title("extracted patch")
    plt.axis('off')  # Hide axes ticks and labels for a cleaner image display
    plt.show()


def extract_patches_camelyon(src_data_df, src_coor_dir, gt_mask_dir, src_wsi_dir, dest_dir, wsi_ext):

    tumor_patch_thred = 10 ##how many tumor pixels are required to consider a patch as a tumor patch.
    patch_size = 512, ## For camelyon data, at patch level 1, the resolution is 20x. this corresponds to 512x512 pixels at level 0.

    ### prepare the destination directory
    neg_dest_dir = os.path.join(dest_dir, "negative_patches")
    pos_dest_dir = os.path.join(dest_dir, "positive_patches")
    if not os.path.exists(neg_dest_dir):
        os.makedirs(neg_dest_dir, exist_ok=True)
    if not os.path.exists(pos_dest_dir):
        os.makedirs(pos_dest_dir, exist_ok=True)

    col_names = ['slide_id', 'label', 'x', 'y', 'w', 'h','level']  # note x and y are at level0, w, h are at level specfied by 'level'
    pos_df = pd.DataFrame(columns=col_names)
    neg_df = pd.DataFrame(columns=col_names)

    tumor_pixels = [2]
    normal_pixels = [1]

    pos_count = 0
    neg_count = 0

    while pos_count <= 10000 or neg_count <= 10000:
        for index, row in src_data_df.iterrows():
            slide_id = row['slide_id']
            feature_file = os.path.join(src_coor_dir, slide_id + ".h5")
            wsi_path = os.path.join(src_wsi_dir, slide_id + wsi_ext)
            if os.path.exists(feature_file) == False or os.path.exists(wsi_path) == False:
                print(f"Feature file or WSI file does not exist for {slide_id}. Skipping...")
                continue

            reader = mir.MultiResolutionImageReader()
            wsi_img = reader.open(wsi_path)

            with h5py.File(feature_file, 'r') as f:
                feats = f['features'][:]
                coords = f['coords'][:]  ## with shape (N, 2), where N is the number of patches, each row is [x, y]

            if row['label'] == 0:  ## only extract negative patches
                ### don't rely on the ground truth mask, just randomly select patches using the coordinates from the features.
                ### randomly sample 10 patches from the coordinates, and extract the patches from the WSI image.
                sample_number = min(len(coords), 50)  ## sample at most 10 patches
                sample_idx = random.sample(range(0, len(coords)), sample_number)
                for i in range(len(sample_idx)):
                    if neg_count <= 5000:   #if neg_count <= 10000:
                        patch_coord = coords[sample_idx[i],:]
                        patch_feature = feats[sample_idx[i],:]
                        x_level0_corner = int(patch_coord[0])
                        y_level0_corner = int(patch_coord[1]) ### x, y are all at level 0, regardless of the resolution used for feature extraction.
                        extracted_patch = wsi_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0)  # level 0 is the highest resolution
                        new_row = {
                            'slide_id': slide_id,
                            'label': row['label'],
                            'x': x_level0_corner,
                            'y': y_level0_corner,
                            'w': str(patch_size),
                            'h': str(patch_size)
                        }
                        plt.imshow(extracted_patch)
                        plt.title("extracted patch")
                        plt.axis('off')  # Hide axes ticks and labels for a cleaner image display
                        plt.show()

                        # file_name = slide_id + "_x_" + str(x_level0_corner) + "_y_" + str(y_level0_corner) + ".png"
                        # dest_path = os.path.join(neg_dest_dir, file_name)
                        # plt.imsave(dest_path, extracted_patch)

                        file_name = slide_id + "_x_" + str(
                            x_level0_corner) + "_y_" + str(
                            y_level0_corner) + "_raw_feat.pt"
                        dest_path = os.path.join(neg_dest_dir, file_name)
                        torch.save(patch_feature, dest_path)

                        # Append the new row to the DataFrame
                        neg_df = neg_df._append(new_row, ignore_index=True)
                        neg_count += 1

            if row['label'] == 1:  ## extract both positive and negative patches, using the ground truth mask.
                pos_per_sample = 200
                neg_per_sample = 50
                gt_mask_file_path = os.path.join(gt_mask_dir, row['slide_id'] + "_mask" + wsi_ext)
                if os.path.exists(gt_mask_file_path) is False:
                    print("ground truth mask file does not exist for {}, skipping...".format(row['slide_id']))
                    continue
                else:
                    reader = mir.MultiResolutionImageReader()
                    mask_img = reader.open(gt_mask_file_path)
                ## read the ground truth mask and select patch according to the mask.
                ## shuffle  the coordinates to avoid only using boundary coordinates.
                sample_idx = random.sample(range(0, len(coords)), len(coords))

                for i in range(len(sample_idx)):
                    if pos_per_sample > 0 or neg_per_sample > 0:  ## mean
                        patch_coord = coords[sample_idx[i], :]
                        patch_feature = feats[sample_idx[i], :]
                        x_level0_corner = int(patch_coord[0])  # if not convert to int, idx[1][sample_idx[i]] is of type numpy.int64, which is not accepted by getUCharPatch
                        y_level0_corner = int(patch_coord[1])
                        patch_mask = mask_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0).squeeze()  # level 0 is the highest resolution
                        tumor_mask = np.isin(patch_mask, tumor_pixels)
                        if np.sum(tumor_mask) > tumor_patch_thred:
                            if pos_count <= 10000 and pos_per_sample > 0:
                                print("tumor patch!")
                                extracted_patch = wsi_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0)
                                # plt.imshow(extracted_patch)
                                # plt.title("extracted patch")
                                # plt.axis('off')
                                new_row = {
                                    'slide_id': slide_id,
                                    'label': row['label'],
                                    'x': x_level0_corner,
                                    'y': y_level0_corner,
                                    'w': str(patch_size),
                                    'h': str(patch_size)
                                }

                                file_name = slide_id + "_x_" + str(x_level0_corner) + "_y_" + str(y_level0_corner) + "_raw_feat.pt"
                                dest_path = os.path.join(pos_dest_dir, file_name)
                                torch.save(patch_feature, dest_path)

                                # Append the new row to the DataFrame
                                pos_df = pos_df._append(new_row, ignore_index=True)
                                pos_count += 1
                                pos_per_sample = pos_per_sample - 1
                        else:
                            if np.sum(tumor_mask) == 0:
                                normal_mask = np.isin(patch_mask, normal_pixels)
                                if neg_count <= 10000 and np.sum(normal_mask) > 5000 and neg_per_sample > 0:
                                    extracted_patch = wsi_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0)
                                    # plt.imshow(extracted_patch)
                                    # plt.title("extracted patch")
                                    # plt.axis('off')
                                    new_row = {
                                        'slide_id': slide_id,
                                        'label': row['label'],
                                        'x': x_level0_corner,
                                        'y': y_level0_corner,
                                        'w': str(patch_size),
                                        'h': str(patch_size)
                                    }

                                    file_name = slide_id + "_x_" + str(x_level0_corner) + "_y_" + str(y_level0_corner) + "_raw_feat.pt"
                                    dest_path = os.path.join(neg_dest_dir,file_name)
                                    torch.save(patch_feature, dest_path)
                                    # Append the new row to the DataFrame
                                    neg_df = neg_df._append(new_row, ignore_index=True)
                                    neg_count += 1
                                    neg_per_sample = neg_per_sample - 1

    pos_df.to_csv(os.path.join(pos_dest_dir, "positive_patches.csv"), index=False)
    neg_df.to_csv(os.path.join(neg_dest_dir, "negative_patches.csv"), index=False)



def extract_patches_panda(src_data_df, src_coor_dir, gt_mask_dir, src_wsi_dir, dest_dir, wsi_ext):
    ### this function we extract negative patches. different from previous one, we directly use the coordinates from the features, instead of extract patch by randomly sample mask. to avoid overlap between patches.

    patch_size = 256

    ### prepare the destination directory
    neg_dest_dir = os.path.join(dest_dir, "negative_patches")
    pos_dest_dir = os.path.join(dest_dir, "positive_patches")
    if not os.path.exists(neg_dest_dir):
        os.makedirs(neg_dest_dir, exist_ok=True)
    if not os.path.exists(pos_dest_dir):
        os.makedirs(pos_dest_dir, exist_ok=True)

    col_names = ['slide_id', 'label', 'x', 'y', 'w', 'h','level']  # note x and y are at level0, w, h are at level specfied by 'level'
    pos_df = pd.DataFrame(columns=col_names)
    neg_df = pd.DataFrame(columns=col_names)

    tumor_pixel_value_dict = {
        "karolinska": [2],
        "radboud": [3, 4, 5],
    }
    normal_pixel_value_dict = {
        "karolinska": [1],
        "radboud": [1, 2],
    }

    pos_count = 0
    neg_count = 0

    for index, row in src_data_df.iterrows():
        if pos_count > 10000 and neg_count > 10000:
            break
        slide_id = row['slide_id']

        feature_file = os.path.join(src_coor_dir, slide_id + ".h5")
        wsi_path = os.path.join(src_wsi_dir, slide_id + wsi_ext)
        if os.path.exists(feature_file) == False or os.path.exists(wsi_path) == False:
            print(f"Feature file or WSI file does not exist for {slide_id}. Skipping...")
            continue

        reader = mir.MultiResolutionImageReader()
        wsi_img = reader.open(wsi_path)

        with h5py.File(feature_file, 'r') as f:
            feats = f['features'][:]  ## with shape (N, D), where N is the number of patches, D is the feature dimension
            coords = f['coords'][:]  ## with shape (N, 2), where N is the number of patches, each row is [x, y]

        if row['label'] == 0:  ## only extract negative patches

            sample_number = min(len(coords), 10)  ## sample at most 10 patches
            sample_idx = random.sample(range(0, len(coords)), sample_number)
            for i in range(len(sample_idx)):
                if neg_count <= 10000:
                    patch_coord = coords[sample_idx[i],:]
                    patch_feature = feats[sample_idx[i],:]
                    x_level0_corner = int(patch_coord[0])  # if not convert to int, idx[1][sample_idx[i]] is of type numpy.int64, which is not accepted by getUCharPatch
                    y_level0_corner = int(patch_coord[1]) ### x, y are all at level 0, regardless of the resolution used for feature extraction.
                    extracted_patch = wsi_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0)  # level 0 is the highest resolution
                    new_row = {
                        'slide_id': slide_id,
                        'label': row['label'],
                        'x': x_level0_corner,
                        'y': y_level0_corner,
                        'w': str(patch_size),
                        'h': str(patch_size)
                    }

                    file_name = slide_id + "_x_" + str(x_level0_corner) + "_y_" + str(y_level0_corner) + "_raw_feat.pt"
                    dest_path = os.path.join(neg_dest_dir, file_name)
                    torch.save(patch_feature, dest_path)

                    # Append the new row to the DataFrame
                    neg_df = neg_df._append(new_row, ignore_index=True)
                    neg_count += 1

        if row['label'] == 1:  ## extract both positive and negative patches, using the ground truth mask.
            pos_per_sample = 25
            neg_per_sample = 10
            gt_mask_file_path = os.path.join(gt_mask_dir, row['slide_id'] + "_mask" + wsi_ext)
            if os.path.exists(gt_mask_file_path) is False:
                continue
            else:
                reader = mir.MultiResolutionImageReader()
                mask_img = reader.open(gt_mask_file_path)
            ## read the ground truth mask and select patch according to the mask.
            tumor_pixels = tumor_pixel_value_dict[row['data_provider']]
            normal_pixels = normal_pixel_value_dict[row['data_provider']]
            #### shuffle  the coordinates to avoid only using boundary coordinates.
            sample_idx = random.sample(range(0, len(coords)), len(coords))
            for i in range(len(sample_idx)):
                if pos_per_sample <= 0 and neg_per_sample <= 0:
                    print("collected enough patches, move to the next sample...")
                    break
                patch_coord = coords[sample_idx[i], :]
                patch_feature = feats[sample_idx[i], :]
                x_level0_corner = int(patch_coord[0])  # if not convert to int, idx[1][sample_idx[i]] is of type numpy.int64, which is not accepted by getUCharPatch
                y_level0_corner = int(patch_coord[1])
                patch_mask = mask_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0).squeeze()  # level 0 is the highest resolution
                tumor_mask = np.isin(patch_mask, tumor_pixels)
                if np.sum(tumor_mask) > 1000:
                    if pos_count <= 10000:
                        # print("tumor patch!")
                        extracted_patch = wsi_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0)
                        # plt.imshow(extracted_patch)
                        # plt.title("extracted patch")
                        # plt.axis('off')
                        new_row = {
                            'slide_id': slide_id,
                            'label': row['label'],
                            'x': x_level0_corner,
                            'y': y_level0_corner,
                            'w': str(patch_size),
                            'h': str(patch_size)
                        }
                        file_name = slide_id + "_x_" + str(x_level0_corner) + "_y_" + str(y_level0_corner) + "_raw_feat.pt"
                        dest_path = os.path.join(pos_dest_dir, file_name)
                        torch.save(patch_feature, dest_path)

                        # Append the new row to the DataFrame
                        pos_df = pos_df._append(new_row, ignore_index=True)
                        pos_count += 1
                        pos_per_sample -= 1
                else:
                    normal_mask = np.isin(patch_mask, normal_pixels)
                    if np.sum(normal_mask) > 10000:
                        if neg_count <= 10000:
                            # print("normal patch!")
                            extracted_patch = wsi_img.getUCharPatch(x_level0_corner, y_level0_corner, patch_size, patch_size, 0)
                            # plt.imshow(extracted_patch)
                            # plt.title("extracted patch")
                            # plt.axis('off')
                            new_row = {
                                'slide_id': slide_id,
                                'label': row['label'],
                                'x': x_level0_corner,
                                'y': y_level0_corner,
                                'w': str(patch_size),
                                'h': str(patch_size)
                            }
                            file_name = slide_id + "_x_" + str(
                                x_level0_corner) + "_y_" + str(
                                y_level0_corner) + "_raw_feat.pt"
                            dest_path = os.path.join(neg_dest_dir, file_name)
                            torch.save(patch_feature, dest_path)

                            # Append the new row to the DataFrame
                            neg_df = neg_df._append(new_row, ignore_index=True)
                            neg_count += 1
                            neg_per_sample -= 1

    pos_df.to_csv(os.path.join(pos_dest_dir, "positive_patches.csv"), index=False)
    neg_df.to_csv(os.path.join(neg_dest_dir, "negative_patches.csv"), index=False)

def main(dataset_dict, dataset, task):

    src_data_df = pd.read_csv(dataset_dict[dataset]["src_data_csv"])
    gt_mask_dir = dataset_dict[dataset]["gt_mask_dir"]
    src_wsi_dir = dataset_dict[dataset]["src_wsi_dir"]
    wsi_ext = dataset_dict[dataset]["wsi_ext"]
    dest_dir = dataset_dict[dataset]["dest_dir"]
    src_coor_dir = dataset_dict[dataset]["src_coor_dir"]

    if task == "extract_patches_panda":
        extract_patches_panda(src_data_df, src_coor_dir, gt_mask_dir, src_wsi_dir, dest_dir, wsi_ext)
    if task == "extract_patches_camelyon16":
        extract_patches_camelyon(src_data_df, src_coor_dir, gt_mask_dir, src_wsi_dir, dest_dir, wsi_ext)





if __name__ == '__main__':
    dataset = "panda"  # or "camelyon16", "panda"
    task = "extract_patches_panda"  ## task can be "get_basic_info", "extract_patches_panda", "extract_patches_camelyon16"

    dataset_dict = {
        "camelyon16": {
            "src_data_csv": "/home/susu/datasets/camelyon_split/split_csv/train.csv",
            "gt_mask_dir": "/media/susu/BA11-1CAC/camelyon16/mask_from_CPG/masks",
            "src_wsi_dir": "/media/susu/BA11-1CAC/camelyon16/WSI",
            "src_coor_dir": "/home/susu/datasets/trident_out_camelyon16/20x_256px_0px_overlap/features_conch_v1",
            "dest_dir": "/home/susu/datasets/camelyon16_patches",
            "sample_id": "test_001",
            "patch_size": 256,
            "wsi_ext": ".tif",
            "patch_level": 1 #### For camelyon data, at patch level 1, the resolution is 20x.
        },
        "panda": {
            "src_data_csv": "/home/susu/datasets/panda_split/split_csv/train.csv",
            "gt_mask_dir": "/media/susu/BA11-1CAC/panda/original_dataset/prostate-cancer-grade-assessment/train_label_masks",
            "src_wsi_dir": "/media/susu/BA11-1CAC/panda/original_dataset/prostate-cancer-grade-assessment/train_images",
            "src_coor_dir": "/home/susu/datasets/trident_out_panda/20x_256px_0px_overlap/features_conch_v1",
            "dest_dir": "/home/susu/datasets/panda_patches",
            "sample_id":"0a0f8e20b1222b69416301444b117678",
            "patch_size": 256,
            "wsi_ext": ".tiff",
            "patch_level": 0 #### For PANDA data, at patch level 0, the resolution is 20x.
        }
    }

    if task == "get_basic_info":
        dirs = dataset_dict[dataset]
        get_basic_info(dataset_name=dataset, wsi_id=dirs["sample_id"], wsi_dir=dirs["src_wsi_dir"], gt_dir=dirs["gt_mask_dir"], wsi_ext= dirs["wsi_ext"])

    else:
        main(dataset_dict, dataset, task)