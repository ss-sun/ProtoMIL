#### I reuse the results generated before using code https://github.com/ss-sun/concept_MIL/blob/main/feature_space_analysis.py
import os
import numpy as np
from sparse_autoencoder import SparseAutoencoder
import torch

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import copy
import sys

os.environ['PYTHONPATH'] = "/opt/ASAP/bin"
# Optionally, add the path to sys.path
sys.path.append("/opt/ASAP/bin")
import multiresolutionimageinterface as mir


def load_sae_model(ckpt_path, autoencoder_input_dim, n_learned_features, device):
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=1).to(device)
    state_dict = torch.load(ckpt_path)
    autoencoder.load_state_dict(state_dict)
    return autoencoder


## Extract sae embeddings
def extract_sae_embeddings(src_file, dest_file, sae_model, device):
    src_data = np.load(src_file)
    src_data = torch.tensor(src_data).to(device)
    concept_embedding, reconstructions = sae_model(src_data)
    concept_embedding = concept_embedding.squeeze().detach().cpu().numpy()
    np.save(dest_file, concept_embedding)
    print(concept_embedding.shape)

def extract_concept_activation(src_img_embed_dir, dest_concept_embed_dir, sae_model, device):
    img_files = [file for file in os.listdir(src_img_embed_dir) if file.endswith('.pt')]
    for img_file in img_files:
        img_activation = torch.load(os.path.join(src_img_embed_dir, img_file))
        img_activation = img_activation.to(device)
        concept_embedding, reconstructions = sae_model(img_activation)
        concept_embedding = concept_embedding.squeeze().detach().cpu()
        dest_file = os.path.join(dest_concept_embed_dir, img_file)
        torch.save(concept_embedding, dest_file)



if __name__ == "__main__":

    '''
    (1) run "inspect sae" to check the loading of model and the dimensions of the embeddings
    (2) run "extract_sae_embeddings" to extract the embeddings from the images using the SAE model. store the embeddings in the model directory. e.g. "/home/susu/sae_exps/SAEImg/all_train/all_train_2025-01-30_17:41_lr0.0001_l1coeff0.0003_ef2_rf500000_hookout_bs4096_epo30/sae/camelyon_positive_patches_concept_features.npy"
    (3) run "select_nonzeros_neurons" to get statics of the non-zero neurons, such as the index of neurons and the activated frequency. The results are stored in a json file. e.g. "/home/susu/sae_exps/SAEImg/all_train/all_train_2025-01-30_17:41_lr0.0001_l1coeff0.0003_ef2_rf500000_hookout_bs4096_epo30/nonzero_neurons.json"
    (4) run "select_top_activated_patches" to get top 10 activated patches for each neuron. The results are stored in a json file. e.g. "/home/susu/sae_exps/SAEImg/all_train/all_train_2025-01-30_17:41_lr0.0001_l1coeff0.0003_ef2_rf500000_hookout_bs4096_epo30/top_activated_patches.json"
    (5) run "prepare_prototypes", to put the top 10 activated patches inside a folder for each neuron.
    (6) run "vis_top_activated_patches" to visualize the top 10 activated patches for each neuron.
    '''

    task = "vis_top_activated_patches" # "get_prototype_imgs" #"prepare_prototypes" # "create_file_list" #"prepare_prototypes" #"select_top_activated_patches" # "select_nonzeros_neurons" # "extract_sae_embeddings" # "extract_sae_embeddings" #"gather_img_features" ## task can be "prepare_prototypes", "inspect_sae","select_top_activated_patches" ,"distribution_of_activations", "extract_sae_embeddings" or "select_nonzeros_neurons", "analysis_top_k"

    ckpt_path = "/home/susu/sae_exps/SAEImg/camelyon_panda/conch/out/camelyon_panda_2025-09-12_14:27_lr0.0001_l1coeff0.0003_ef4_rf500000_hookout_bs4096_epo1/sae_checkpoints/sparse_autoencoder_final.pt"
    efactor = 4
    src_data_dirs = {
        "camelyon": "/home/susu/datasets/camelyon16_patches",
        "panda": "/home/susu/datasets/panda_patches",
    }

    if task == "inspect_sae":
        autoencoder_input_dim = 512
        n_learned_features = autoencoder_input_dim * efactor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sae_model = load_sae_model(ckpt_path, autoencoder_input_dim, n_learned_features, device)
        print(sae_model)

        # src_img_feature_npy = os.path.join(src_data_dirs["camelyon"], f"positive_patches_img_features.npy")
        # img_embd = np.load(src_img_feature_npy)
        # print("img_embd.shape: ", img_embd.shape)
        # sample_img_embd = img_embd[0,:]
        # sample_img_embd = torch.tensor(sample_img_embd).to(device)
        # concept_embedding, reconstructions = sae_model(sample_img_embd)
        # print("concept_embedding.shape: ", concept_embedding.shape)
        # print("torch.max(concept_embedding): ", torch.max(concept_embedding))
        # print("torch.min(concept_embedding): ", torch.min(concept_embedding))

    if task == "gather_img_features":
        for dataset in ["panda", "camelyon"]:
            for folder_name in ["positive_patches", "negative_patches"]:
                tgt_folder = src_data_dirs[dataset]
                src_folder = os.path.join(tgt_folder, folder_name)
                dest_file = os.path.join(tgt_folder, f"{folder_name}_normalized_img_features.npy")
                dest_array = []
                all_features = [f for f in os.listdir(src_folder) if "_raw_feat.pt" in f] ## raw_feat mean unnormalized features
                all_features = sorted(all_features)
                for feat in all_features:
                    src_file = os.path.join(src_folder, feat)
                    img_feature = torch.load(src_file)
                    l2_norm = np.linalg.norm(img_feature)
                    img_feature = img_feature / l2_norm
                    dest_array.append(img_feature)
                dest_array = np.array(dest_array)
                np.save(dest_file, dest_array)

    if task == "create_file_list":
        for dataset in ["panda", "camelyon"]:
            all_files = []
            for folder_name in ["positive_patches", "negative_patches"]:
                tgt_folder = src_data_dirs[dataset]
                src_folder = os.path.join(tgt_folder, folder_name)
                all_features = [f for f in os.listdir(src_folder) if "_raw_feat.pt" in f]  ## raw_feat mean unnormalized features
                all_features = sorted(all_features)
                all_files.append(all_features)
            flattened = [item for sublist in all_files for item in sublist]
            dest_folder = os.path.join(src_data_dirs[dataset], "pos_neg_file_list")
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            patch_files_path = os.path.join(dest_folder, f"{dataset}_patches_file_names.txt")
            ## write the file names to a text file
            with open(patch_files_path, "w") as f:
                for file in flattened:
                    f.write(file + "\n")

    if task == "extract_sae_embeddings": ## save the sae embeddings at location such as /home/susu/datasets/panda_patches/sae/positive_patches_concept_features.npy
        autoencoder_input_dim = 512
        n_learned_features = autoencoder_input_dim * efactor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sae_model = load_sae_model(ckpt_path, autoencoder_input_dim, n_learned_features, device)
        for dataset in ["panda", "camelyon"]:
            tgt_folder = src_data_dirs[dataset]
            for label in ["positive", "negative"]:
                src_img_feature_npy = os.path.join(tgt_folder, f"{label}_patches_normalized_img_features.npy")
                img_embd = np.load(src_img_feature_npy)
                print("img_embd.shape: ", img_embd.shape)
                # dest_dir = os.path.join(tgt_folder, "sae") ## "/home/susu/datasets/panda_patches/sae"
                model_dir = os.path.dirname(ckpt_path)
                dest_dir = os.path.join(model_dir, "probing_imgs")
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                dest_file = os.path.join(dest_dir, f"{dataset}_{label}_patches_concept_features.npy")
                print("dest_file: ", dest_file)
                sae_embd = extract_sae_embeddings(src_img_feature_npy, dest_file, sae_model, device)
                print("SAE embeddings saved at: ", dest_file)


    if task == "select_nonzeros_neurons":

        src_dir = os.path.join(os.path.dirname(ckpt_path), "probing_imgs")
        src_files = [file for file in os.listdir(src_dir) if file.endswith('.npy') and "concept_features" in file]

        results = {}
        for dataset in ["camelyon", "panda"]:
            results[dataset] = {}

        for file in src_files:
            src_file = os.path.join(src_dir, file)
            dataset = file.split("_")[0] # e.g., "camelyon", "panda"
            label = file.split("_")[1] # e.g., "positive", "negative"
            print("src_file: ", src_file)
            data = np.load(src_file)
            nonzero_indices_per_row = [np.where(row != 0)[0] for row in data]
            flat_list = [item for sublist in nonzero_indices_per_row for item in sublist]
            unique_indices, count = np.unique(flat_list, return_counts=True)

            idx_result = {}
            print("unique_indices: ",unique_indices)
            print("count: ", count)
            print("len(unique_indices): ", len(unique_indices))
            idx_result["unique_indices"] = unique_indices.tolist()
            idx_result["count"] = count.tolist()
            idx_result["len_unique_indices"] = len(unique_indices)
            results[dataset][label] = idx_result
        dest_file = os.path.dirname(ckpt_path) + "/nonzero_neurons.json"
        with open(dest_file, "w") as f:
            json.dump(results, f)

    if task == "select_top_activated_patches":

        nonzero_neuron_file = os.path.dirname(ckpt_path) + "/nonzero_neurons.json"
        with open(nonzero_neuron_file) as f:
            nonzero_neurons = json.load(f)
        results = copy.deepcopy(nonzero_neurons)

        for dataset in ["camelyon", "panda"]:
            all_concept_features = []
            positive_file = os.path.join(os.path.dirname(ckpt_path), "probing_imgs", f"{dataset}_positive_patches_concept_features.npy")
            positive_features = np.load(positive_file)
            negative_file = os.path.join(os.path.dirname(ckpt_path), "probing_imgs", f"{dataset}_negative_patches_concept_features.npy")
            negative_features = np.load(negative_file)
            ## put two features together.
            all_feature = np.concatenate((positive_features, negative_features), axis=0)
            print(positive_features.shape)
            print(negative_features.shape)
            print(all_feature.shape)
            nonzero_idx_positive = nonzero_neurons[dataset]["positive"]["unique_indices"]
            nonzero_idx_negative = nonzero_neurons[dataset]["negative"]["unique_indices"]
            nonzero_idx = np.unique(np.concatenate((nonzero_idx_positive, nonzero_idx_negative)))
            results[dataset]["unique_indices"] = nonzero_idx.tolist()
            print("nonzero_idx: ", nonzero_idx)
            print(len(nonzero_idx))
            for idx in nonzero_idx:
                idx = int(idx)
                results[dataset][idx] = {}
                concept_activations = all_feature[:, idx].squeeze()
                total_activations = np.sum(concept_activations)
                highest_10_activations_idx = np.argsort(concept_activations)[::-1][:10]
                highest_10_activations = concept_activations[highest_10_activations_idx]
                results[dataset][idx]["total_activations"] = float(total_activations)
                results[dataset][idx]["highest_10_activations_idx"] = highest_10_activations_idx.tolist()
                results[dataset][idx]["highest_10_activations"] = highest_10_activations.tolist()
        results_file = os.path.dirname(ckpt_path) + "/top_activated_patches_pos_and_neg.json"
        with open(results_file, "w") as f:
            json.dump(results, f)



    if task == "prepare_prototypes":
        dest_folder = os.path.dirname(ckpt_path) + "/proto_types"
        with open(os.path.dirname(ckpt_path) + "/top_activated_patches_pos_and_neg.json") as f:
            top_activated_patches = json.load(f)
        for dataset in ["camelyon", "panda"]:
            # patch_size = patch_size_dict[dataset]
            output_dir = os.path.join(dest_folder, dataset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            patch_files_path = os.path.join(src_data_dirs[model_key][dataset],"pos_neg_file_list", f"{dataset}_patches_file_names.txt")
            with open(patch_files_path, "r") as f:
                patch_files = f.readlines()
                patch_files = [file.strip() for file in patch_files]
            print("patch_files: ", patch_files[0])

            # src_img_foler = os.path.join(src_data_dirs[dataset], f"{label}_patches")
            # patch_files = [file for file in os.listdir(src_img_foler) if file.endswith('.png')]
            # patch_files = sorted(patch_files)
            unique_indices = top_activated_patches[dataset]["unique_indices"]
            print("unique_indices: ", unique_indices)
            print("len(unique_indices): ", len(unique_indices))
            for neuron_idx in unique_indices:
                print("neuron_idx: ", neuron_idx)
                patches = top_activated_patches[dataset][str(neuron_idx)]["highest_10_activations_idx"]
                activations_values = top_activated_patches[dataset][str(neuron_idx)]["highest_10_activations"]
                print("patches: ", patches)
                proto_output_dir = os.path.join(output_dir, str(neuron_idx))
                if not os.path.exists(proto_output_dir):
                    os.makedirs(proto_output_dir, exist_ok=True)
                for i in range(len(patches)):
                    patch_idx = patches[i]
                    patch_file = patch_files[patch_idx]
                    print("patch_file: ", patch_file)
                    activations_value = activations_values[i]
                    print("activations_value: ", activations_value)
                    src_img_folers = [os.path.join(src_data_dirs[model_key][dataset], f"{label}_patches") for label in ["positive", "negative"]]
                    for src_img_foler in src_img_folers:
                        src_patch_file = os.path.join(src_img_foler, patch_file)
                        if os.path.exists(src_patch_file):
                            break
                    # src_patch_file = os.path.join(src_img_foler, patch_file)
                    dest_patch_file = os.path.join(proto_output_dir, patch_file)
                    os.system(f"cp {src_patch_file} {dest_patch_file}")


    if task == "get_prototype_imgs":

        patch_size_dict = {
            "camelyon": {
                "20x": 512,
            },
            "panda":{
                "20x": 256,
            }
        }

        for model_key in ckpt_path_dict.keys():
            src_root_foler = os.path.dirname(ckpt_path_dict[model_key]) + "/proto_types"
            dest_root_foler = os.path.dirname(ckpt_path_dict[model_key]) + "/proto_types_imgs"
            if not os.path.exists(dest_root_foler):
                os.makedirs(dest_root_foler, exist_ok=True)
            for dataset in ["camelyon", "panda"]:
                # patch_size = patch_size_dict[dataset][model_key]

                dest_dir = os.path.join(dest_root_foler, dataset)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                if dataset == "camelyon":
                    wsi_dir = "/media/susu/BA11-1CAC/camelyon16/WSI"
                    wsi_ext = ".tif"
                if dataset == "panda":
                    wsi_dir = "/media/susu/BA11-1CAC/panda/original_dataset/prostate-cancer-grade-assessment/train_images"
                    wsi_ext = ".tiff"
                src_foler = os.path.join(src_root_foler, dataset)
                all_neuron_folders = [f for f in os.listdir(src_foler) if os.path.isdir(os.path.join(src_foler, f))]
                for neuron_folder in all_neuron_folders:
                    prototypes = os.listdir(os.path.join(src_foler, neuron_folder))
                    for prototype in prototypes:
                        if dataset == "camelyon":
                            file_names = prototype.split("_")
                            slide_id = file_names[0]+ "_" + file_names[1]
                            x_coord = file_names[3]
                            y_coord = file_names[5]
                        if dataset == "panda":
                            file_names = prototype.split("_")
                            slide_id = file_names[0]
                            x_coord = file_names[2]
                            y_coord = file_names[4]

                        wsi_path = os.path.join(wsi_dir, slide_id + wsi_ext)
                        reader = mir.MultiResolutionImageReader()
                        wsi_img = reader.open(wsi_path)

                        x_level0_corner = int(x_coord)  # if not convert to int, idx[1][sample_idx[i]] is of type numpy.int64, which is not accepted by getUCharPatch
                        y_level0_corner = int(y_coord)  ### x, y are all at level 0, regardless of the resolution used for feature extraction.
                        extracted_patch = wsi_img.getUCharPatch(x_level0_corner,
                                                                y_level0_corner,
                                                                patch_size,
                                                                patch_size,
                                                                0)  # level 0 is the highest resolution

                        output_dir = os.path.join(dest_dir, neuron_folder)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        file_name = slide_id + "_x_" + str(x_level0_corner) + "_y_" + str(y_level0_corner) + ".png"
                        dest_path = os.path.join(output_dir, file_name)
                        plt.imsave(dest_path, extracted_patch)






    if task == "vis_top_activated_patches":
        plot_individual = False

        # with open("/home/susu/datasets/named_neurons/sae/top_activated_patches.json") as f:
        #     top_activated_patches = json.load(f)
        # with open(os.path.dirname(ckpt_path_dict[model_key]) + "/top_activated_patches.json") as f:
        #     top_activated_patches = json.load(f)
        for model_key in ckpt_path_dict.keys():
            with open(os.path.dirname(ckpt_path_dict[model_key]) + "/top_activated_patches_pos_and_neg.json") as f:
                top_activated_patches = json.load(f)
            for dataset in ["camelyon", "panda"]:
                # for label in ["positive", "negative"]:

                # src_img_foler = os.path.join(src_data_dirs[dataset], f"{label}_patches")
                # patch_files = [file for file in os.listdir(src_img_foler) if file.endswith('.png')]
                # patch_files = sorted(patch_files)

                # patch_files_path = os.path.join(src_data_dirs[model_key][dataset], "pos_neg_file_list", f"{dataset}_patches_file_names.txt")
                # with open(patch_files_path, "r") as f:
                #     patch_files = f.readlines()
                #     patch_files = [file.strip() for file in patch_files]
                # print("patch_files: ", patch_files[0])
                # src_img_folers = [os.path.join(src_data_dirs[dataset], f"{label}_patches") for label in ["positive", "negative"]]

                count = top_activated_patches[dataset]["count"]
                unique_indices = top_activated_patches[dataset]["unique_indices"]
                ## sort the unique_indices based on the count
                sorted_indices = [x for _, x in sorted(zip(count, unique_indices), reverse=True)]
                most_freq_neuron_indices = sorted_indices[:10]
                print("sorted_indices: ", sorted(most_freq_neuron_indices))

                neuron_idx = []
                top_images = []
                top_image_activations = []
                for index in most_freq_neuron_indices:
                    index = str(index)
                    selected_patches = top_activated_patches[dataset][index]["highest_10_activations_idx"]
                    patch_activations = top_activated_patches[dataset][index]["highest_10_activations"]
                    # print("selected_patches: ", selected_patches)
                    # print("patch_activations: ", patch_activations)
                    imgs = []
                    for idx in selected_patches:
                        patch_file = patch_files[idx]
                        for src_img_foler in src_img_folers:
                            img_file = os.path.join(src_img_foler, patch_file)
                            if os.path.exists(img_file):
                                break
                        # img_file = os.path.join(src_img_foler, patch_file)
                        patch = plt.imread(img_file)
                        imgs.append(patch)

                    if plot_individual == True:
                        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                        for i, ax in enumerate(axes.flat):
                            ax.imshow(imgs[i])
                            ax.set_title(f"Activation: {patch_activations[i]:.2f}")
                            ax.axis("off")
                            # set the title of the plot
                        title = f"neuron {index}: top 10 activated patches on {label} class in {dataset} dataset"
                        fig.suptitle(title)
                        # save the plot
                        plt.tight_layout()
                        output_path = os.path.dirname(ckpt_path_dict[model_key]) + f"/{dataset}_{label}_{index}.png"
                        plt.savefig(output_path)
                        # plt.savefig(f"/home/susu/datasets/named_neurons/sae/vis_top_activated_patches/{dataset}_{label}_{index}.png")
                        plt.close()
                    else:
                        top_images.append(imgs)
                    neuron_idx.append(index)
                    top_image_activations.append(patch_activations)

                fig, axes = plt.subplots(10, 10, figsize=(20, 26))
                for i, ax in enumerate(axes[:, 0]):
                    ax.set_ylabel(f"Neuron {neuron_idx[i]}", fontweight='bold')
                for i in range(10):
                    for j in range(10):
                        axes[i, j].imshow(top_images[i][j])
                        axes[i, j].set_title(f"Activation: {top_image_activations[i][j]:.2f}")
                        # axes[i, j].axis("off")
                title = f"Top 10 activated patches on {label} class in {dataset} dataset"
                fig.suptitle(title, fontsize=20, fontweight='bold')
                plt.tight_layout(rect=[0.05, 0, 1, 0.975])
                output_path = os.path.dirname(ckpt_path_dict[model_key]) + f"/{dataset}_{label}.png"
                # plt.savefig(f"/home/susu/datasets/named_neurons/sae/vis_top_activated_patches/{dataset}_{label}.png")
                plt.savefig(output_path)


    if task == "analysis_nonzeros":
        neuron_names_csv = "/home/susu/datasets/named_neurons/compare_palga_pathtext.csv"
        df_neuron = pd.read_csv(neuron_names_csv)
        with open("/home/susu/datasets/named_neurons/sae/nonzero_neurons.json") as f:
            results = json.load(f)
        # Define the column names
        columns = ['unique_indices', 'count', 'palga_name', 'path_text name']
        for key in ["camelyon", "panda"]:
            for label in ["positive", "negative"]:
                df_out = pd.DataFrame(columns=columns)
                unique_indices = results[key][label]["unique_indices"]
                count = results[key][label]["count"]
                print("unique_indices: ", unique_indices)
                print("count: ", count)
                print(len(unique_indices))
                print(len(count))
                for i in range(len(unique_indices)):
                    idx = unique_indices[i]
                    count_i = count[i]
                    print("idx: ", idx)
                    row = df_neuron.iloc[idx]
                    palga_name = row["palga_name"]
                    path_text_name = row["path_text name"]
                    df_out = df_out._append({"unique_indices": idx, "count": count_i, "palga_name": palga_name, "path_text name": path_text_name}, ignore_index=True)
                with open(f"/home/susu/datasets/named_neurons/sae/nonzero_neurons_{key}_{label}.csv", "w") as f:
                    df_out.to_csv(f, index=False)

    if task =="create_filename_list":
        for dataset in ["camelyon", "panda"]:
            file_names = []
            for label in ["positive", "negative"]:
                src_img_foler = os.path.join(src_data_dirs[dataset], f"{label}_patches")
                patch_files = [file for file in os.listdir(src_img_foler) if file.endswith('.png')]
                patch_files = sorted(patch_files)
                output_file = os.path.join(src_data_dirs[dataset], "pos_neg_file_list", f"{label}_patches_file_names.txt")
                with open(output_file, "w") as f:
                    for file in patch_files:
                        f.write(file + "\n")
                file_names.extend(patch_files)
            output_file = os.path.join(src_data_dirs[dataset], "pos_neg_file_list", f"{dataset}_patches_file_names.txt")
            with open(output_file, "w") as f:
                for file in file_names:
                    f.write(file + "\n")








