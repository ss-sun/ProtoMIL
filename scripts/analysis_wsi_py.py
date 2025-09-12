import os
import argparse
import torch
import shutil
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import DataLoader
from datasets.wsi_dataset import WSIEmbeddingDataset
from models.ProtoMIL import ProtoMIL
import matplotlib
matplotlib.use('Agg')  # Example for non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
# import wsi_core.WholeSlideImage
from wsi_core.WholeSlideImage import WholeSlideImage
from tqdm import tqdm
# from wsi_core.WholeSlideImage import WholeSlideImage

color_dict = {
    'tumor': "#FF7F0E",
    'normal': "#1F77B4",
}
exp_folder = "/home/susu/sae_exps/MLcloud"

exp_names = {
    "camelyon": {
        "img": "proto_mil_camelyon_feat:image_2025-01-16_16:25_lr0.0001_epo200",
        "concept": "proto_mil_camelyon_feat:concept_2025-02-10_09:38_lr0.0001_epo200",
        "perturbed_concept": "seed:12_proto_mil_camelyon_feat:concept_2025-02-27_14:26_lr0.0001_epo200_with_perturbed_neuron_customized",
    },
    "panda": {
        "img": "proto_mil_panda_feat:image_2025-01-16_16:25_lr0.0001_epo200",
        "concept": "proto_mil_panda_feat:concept_2025-02-10_09:38_lr0.0001_epo200",
        "perturbed_concept": "proto_mil_panda_feat:concept_2025-02-19_14:53_lr0.0001_epo200_with_perturbed_neuron_customized",
    },
}


def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)

    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap


def plot_attention_map(A_patch, coords, topk_coords, wsi_path, output_dir, prefix, patch_size=512, vis_level=5, cmap="jet"):
    # vis_level=5, downsample ratio = 32
    # with patch_size=256 at level 0 --> patch size = 8 at level 5
    # vis_patch_size = 8
    # scores = A_patch.detach().cpu()
    heatmap = drawHeatmap(scores=A_patch, coords=coords, slide_path=wsi_path, wsi_object=None, cmap=cmap,
                          alpha=0.4, use_holes=True, binarize=False, vis_level=vis_level, blank_canvas=True,
                          thresh=-1, patch_size=patch_size, convert_to_percentiles=True, segment=False)


    print("heatmap.size: ", heatmap.size)
    draw = ImageDraw.Draw(heatmap)
    # for coord in topk_coords:
    #     # Define the rectangle's top-left and bottom-right coordinates
    #     if vis_level == 5:
    #         rect_top_left = (coord[0]//32, coord[1]//32) # use 32 because the downsample ratio is 32
    #         rect_bottom_right = (coord[0]//32 + 16, coord[1]//32 + 16) # use 8 because the after downsample, 256 patch at level1 becomes 512 patch at level 0, 512 patch becomes to 8
    #         print("rect_top_left", rect_top_left)
    #         print("rect_bottom_right", rect_bottom_right)
    #     if vis_level == 2:
    #         rect_top_left = (coord[0]//16, coord[1]//16) # use 32 because the downsample ratio is 32
    #         rect_bottom_right = (coord[0]//16 + 16, coord[1]//16 + 16) # use 8 because the after downsample, 256 patch becomes to 8
    #
    #     # Draw the rectangle (outline only)
    #     draw.rectangle([rect_top_left, rect_bottom_right], fill='green')
    heatmap.save(os.path.join(output_dir, f'{prefix}.png'))
    plt.axis('off')
    plt.imshow(heatmap)



def load_model(args, device):
    # load ProtoMIl model
    input_dims = args.autoencoder_input_dim * args.expansion_factor
    model = ProtoMIL(input_dims=input_dims, output_dims=args.num_classes, which_version="v1")
    model_path = os.path.join(exp_folder, exp_names[args.probe_dataset][args.feature_type], "last_model.pth")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model
    # print(model)

# load WSI concept activations
def init_dataloaders(args):
    if args.task == "global_expl":
        df = pd.read_csv(os.path.join(args.src_folder, "train.csv"))

    if args.feature_type == 'concept':
        data_source = args.concept_embeddings_dir
        dataset = WSIEmbeddingDataset(data_source=data_source,
                                           df=df,
                                           index_col="slide_id",
                                           target_col="label",
                                           perturbed_neuron=[])

    if args.feature_type == 'perturbed_concept':
        data_source = args.concept_embeddings_dir
        if args.probe_dataset == 'camelyon':
            perturbed_neuron_list = [267, 439, 585, 908, 1509, 1693]  ## adapted to camelyon

        if args.probe_dataset == 'panda':
            perturbed_neuron_list = [267, 339, 439, 585, 908, 1509, 1568]  ## adapted to panda

        dataset = WSIEmbeddingDataset(data_source=data_source,
                                           df=df,
                                           index_col="slide_id",
                                           target_col="label",
                                           perturbed_neuron=perturbed_neuron_list)

    # if args.feature_type == 'image':
    #     data_source = args.activation_dir

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    return dataloader



def vis_wsi_concept_contributions(wsi_concept_contributions, output_dir, prefix, title, add_sum_caption=True):

    # Sample data
    concepts = list(wsi_concept_contributions.keys())
    contributions = list(wsi_concept_contributions.values())
    sum_contributions = np.sum(contributions)

    if add_sum_caption:
        labels = [f"{value:+.2f}" for value in contributions]  # Adding + for positive and - for negative values
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        #labels = [f"{value:.5f}" for value in contributions]  # Adding + for positive and - for negative values
        labels = []
        for value in contributions:
            if value < 0.01:
                label = f"{value:.5f}"
            else:
                label = f"{value:.2f}"
            labels.append(label)
        fig, ax = plt.subplots(figsize=(5.5, 4))

    min_value = min(contributions) * 0.9
    max_value = max(contributions) * 1.1

    # colors = [color_dict['tumor'] if value > 0 else color_dict['normal'] for value in contributions]
    colors = [color_dict['tumor'] if value > 0 else color_dict['normal'] for value in contributions]


    bars = ax.barh(concepts, contributions, color=colors)

    # Adding the labels to the bars
    for bar, label, color in zip(bars, labels, colors):
        ax.text(bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                label,
                va='center',
                ha='left' if bar.get_width() > 0 else 'right',
                color=color,
                fontsize=10)


    # Set x-axis limits to slightly larger than the range of contributions
    ax.set_xlim(min_value, max_value)  # Modify these values to adjust the range as needed

    if title != '':
        ax.set_title(title, fontsize=10)
    if add_sum_caption:
        ax.set_xlabel('Top 10 neuron contributions with sum: {:.2f}'.format(sum_contributions))
    else:
        ax.set_xlabel('Top 10 neuron contributions')

    # Adding gridlines for reference
    ax.axvline(0, color='black', linewidth=1)  # Add a vertical line at x=0

    # Display the plot
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_dir, f'{prefix}.png'))
    plt.close()


def get_global_explanations(model, dataloader, output_dir, neuron_proto_types_dir, device):
    model.eval()

    weights = model.classifier.weight.detach()
    bias = model.classifier.bias.detach()
    # preds = []
    pos_contributions = []
    neg_contributions = []
    logits = []
    for idx, data in enumerate(tqdm(dataloader)):

        # if idx > 50:
        #     break

        wsi_features = data['feature'].to(device).squeeze(0)
        gt_label = data['label'].item()
        results_dict = model(wsi_features)
        bag_logits = results_dict['bag_logits']
        prob = results_dict['Y_prob']
        pred = results_dict["Y_hat"].detach().cpu().numpy().item()
        attn = results_dict["attn"]
        weighted_feat = attn * wsi_features
        pooled_feat = torch.sum(weighted_feat, dim=0, keepdim=True)
        concept_contributions = pooled_feat * weights
        concept_contributions = concept_contributions.detach().cpu().numpy().squeeze()
        if pred == gt_label:
            if gt_label == 1:
                pos_contributions.append(concept_contributions[1])
            if gt_label == 0:
                neg_contributions.append(concept_contributions[0])

    mean_pos_contributions = np.mean(pos_contributions, axis=0)
    mean_neg_contributions = np.mean(neg_contributions, axis=0)

    print("mean_pos_contributions")
    print(mean_pos_contributions)
    print("mean_neg_contributions")
    print(mean_neg_contributions)

    for c_contributions, label in zip([mean_pos_contributions, mean_neg_contributions], ["tumor", "normal"]):
        ### get the top 10 concepts
        c_contributions = torch.tensor(c_contributions)
        top_k_concepts = torch.topk(c_contributions, 20)
        print(top_k_concepts)
        print(top_k_concepts.indices)
        print(top_k_concepts.values)
        wsi_concept_contributions = {}

        for i in range(10):
            idx = top_k_concepts.indices[i].detach().cpu().numpy().item()
            key = f"Neuron_{idx}"
            value = top_k_concepts.values[i].detach().cpu().numpy().item()
            wsi_concept_contributions[key] = value
        print("wsi_concept_contributions")
        print(wsi_concept_contributions)
        vis_wsi_concept_contributions(wsi_concept_contributions, output_dir, f"global_explan_{label}", title='', add_sum_caption=False)

        top_k_neuron_indices = top_k_concepts.indices.detach().cpu().numpy()
        proto_out_dir = os.path.join(output_dir, f"global_explan_{label}")
        if not os.path.exists(proto_out_dir):
            os.makedirs(proto_out_dir, exist_ok=True)
        print(label)
        print(top_k_neuron_indices)
        select_prototyples(top_k_neuron_indices, proto_out_dir, neuron_proto_types_dir)



# make predictions.
def make_predictions(model, img_idx, data_source, data_h5_dir, WSI_dir, output_dir, device, vis_level=5, extension=".tif"):
    model.eval()
    wsi_features = torch.load(os.path.join(data_source, f"{img_idx}.pt")).to(device)
    # print(feat.shape) ## torch.Size([17244, 2048])
    results_dict = model(wsi_features)
    print(results_dict)
    bag_logits = results_dict["bag_logits"]
    pred = results_dict["Y_hat"]
    prob = results_dict["Y_prob"]
    attn = results_dict["attn"]

    # attn = attn.squeeze().detach().cpu().numpy() ##
    weights = model.classifier.weight.detach()
    bias = model.classifier.bias.detach()
    # print(attn.shape) ## (17244,)
    # print(weights.shape) ## (2, 2048)
    # print(bias.shape) ## (2,)

    weighted_feat = attn * wsi_features
    pooled_feat = torch.sum(weighted_feat, dim=0, keepdim=True)
    print(pooled_feat.shape) ## torch.Size([1, 2048])
    print("outside the model")
    print(torch.nonzero(pooled_feat).shape) ## torch.Size([64, 2])
    # concept_contributions = torch.matmul(pooled_feat, weights.T) + bias
    # print(concept_contributions)
    concept_contributions = pooled_feat * weights
    print(concept_contributions.shape)
    # print(torch.sum(concept_contributions, dim=1) + bias)

    if pred == 1:
        c_contributions = concept_contributions[1]
        c_bias = bias[1]
        title = f"Probability of tumor: {prob[0, 1].item():+.2f} - Logit: {bag_logits[0, 1].item():+.2f} - Bias: {c_bias.item():+.2f}"
    if pred == 0:
        c_contributions = concept_contributions[0]
        c_bias = bias[0]
        title = f"Probability of normal: {prob[0, 0].item():+.2f} - Logit: {bag_logits[0, 0].item():+.2f} - Bias: {c_bias.item():+.2f}"
    print(torch.max(c_contributions))
    print(torch.sum(c_contributions))
    print("bias", c_bias)
    ### get the top 10 concepts
    top_k_concepts = torch.topk(c_contributions, 10)
    print(top_k_concepts)
    print(top_k_concepts.indices)
    print(top_k_concepts.values)
    print("sum of the contributions of top_k_concepts",torch.sum(top_k_concepts.values))
    wsi_concept_contributions = {}

    for i in range(10):
        idx = top_k_concepts.indices[i].detach().cpu().numpy().item()
        key = f"Neuron_{idx}"
        value = top_k_concepts.values[i].detach().cpu().numpy().item()
        wsi_concept_contributions[key] = value
    print("wsi_concept_contributions")
    print(wsi_concept_contributions)
    vis_wsi_concept_contributions(wsi_concept_contributions, output_dir, img_idx, title)


    ######### visualize attention map for deep features ##############
    with h5py.File(os.path.join(data_h5_dir, f"{img_idx}.h5"), 'r') as f:
        coords = f['coords'][:]
    wsi_path = os.path.join(WSI_dir, f"{img_idx}{extension}")


    patch_size = 512  ## if dataset is Camelyon, patch size if 512, vis_level=5. if panda, patch size is 256, vis_level=2.


    attn = attn.squeeze().detach().cpu()
    prefix = f"{img_idx}_{str(pred)}_AttnMap"
    topk_coords = []
    plot_attention_map(attn, coords, topk_coords, wsi_path, output_dir, prefix=prefix,
                       patch_size=patch_size, vis_level=vis_level, cmap="coolwarm")


    return pred, top_k_concepts.indices.detach().cpu().numpy()


def select_prototyples(top_k_neuron_indices, output_dir, neuron_proto_types_dir):
    for idx in top_k_neuron_indices:
        neuron_proto_types = os.path.join(neuron_proto_types_dir, str(idx))
        ### copy the image with name "image_layout.jpg" to the output_dir
        img_path = os.path.join(neuron_proto_types, "image_layout.jpg")
        output_path = os.path.join(output_dir, f"neuron_{idx}.jpg")
        # Copy the image to the new location
        shutil.copy(img_path, output_path)
        print(f"Image copied to {output_path}")






# calculate the concept contributions.

# get the top k concepts.

# get the most activated patches on the top K neurons.


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which_machine", type=str, default="workstation", choices=["workstation", "mlcloud", "laptop"])
    parser.add_argument("--probe_dataset", type=str, default="camelyon", choices=["wbcattr", "camelyon", "panda"])
    parser.add_argument("--feature_type", type=str, default='concept', choices=["image", "concept", "perturbed_concept"])
    parser.add_argument("--task", type=str, default='local_expl', choices=["local_expl", "global_expl"])
    parser.add_argument("--autoencoder_input_dim", type=int, default=512,
                        help="input dimension of the autoencoder, we use conch, input dimension is 512")
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes in the dataset, wbcattr is 5, camelyon and panda is 2")

    return parser


if __name__ == "__main__":

    from data_dir_dict import data_dir_dict
    proto_types_dir = "/home/susu/sae_exps/SAEImg/all_train/all_train_2025-01-30_17:41_lr0.0001_l1coeff0.0003_ef4_rf500000_hookout_bs4096_epo30/proto_types"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = get_parser()
    args = parser.parse_args()
    model = load_model(args, device)
    model_dir = os.path.join(exp_folder, exp_names[args.probe_dataset][args.feature_type])
    data_dirs = data_dir_dict[args.which_machine]
    args.src_folder = data_dirs["probe_dataset"][args.probe_dataset]["src_folder"]
    neuron_proto_types_dir = os.path.join(proto_types_dir, args.probe_dataset)

    if args.task == "local_expl":
        img_idx = "test_001" #### tumor case
        # img_idx = "test_071" ###  tumor case
        # img_idx = "test_111"  ### normal case
        ## img_idx = "test_111"  #### normal case

        output_dir = os.path.join(model_dir, "model_output", img_idx)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)


        test_df = pd.read_csv(os.path.join(args.src_folder, "test.csv"))

        if args.feature_type == 'concept':
            data_source = data_dirs["probe_dataset"][args.probe_dataset]["concept_embeddings"]
        if args.feature_type == 'image':
            data_source = data_dirs["probe_dataset"][args.probe_dataset]["activation_dir"]

        data_h5_dir = data_dirs["probe_dataset"][args.probe_dataset]["data_h5_dir"]
        WSI_dir = data_dirs["probe_dataset"][args.probe_dataset]["data_slide_dir"]

        pred, top_k_neuron_indices = make_predictions(model, img_idx, data_source, data_h5_dir, WSI_dir, output_dir,
                                                      device)

        select_prototyples(top_k_neuron_indices, output_dir, neuron_proto_types_dir)


    if args.task == "global_expl":
        args.concept_embeddings_dir = data_dirs["probe_dataset"][args.probe_dataset]["concept_embeddings"]
        output_dir = os.path.join(model_dir, "model_output", "global_expl")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        dataloader = init_dataloaders(args)
        get_global_explanations(model, dataloader, output_dir, neuron_proto_types_dir, device)









