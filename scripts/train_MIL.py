import os
import argparse
import torch
import numpy as np
import random
import datetime
import wandb
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

# from conch.downstream.wsi_datasets import WSIEmbeddingDataset
from models.ProtoMIL import ProtoMIL
from models.ABMIL import DefaultMILGraph
from data_dir_dict import data_dir_dict
from datasets.wsi_dataset import WSIEmbeddingDataset



class ImageWiseCoverageEnergyPercent(torchmetrics.Metric):
    """
    Metric to compute, per image, how many concepts are needed to reach energy_percent of the original logit value.
    Concepts are selected greedily based on contribution to logit.
    Two modes are provided: Contribution to absolute value of the logit, and contribution to the positive part of the logit.
    """

    def __init__(self, energy_percent, use_ground_truth=False,
                 use_only_positive=True):
        super().__init__()
        self.energy_percent = energy_percent
        self.use_ground_truth = use_ground_truth
        self.use_only_positive = use_only_positive
        self.add_state("coverage", default=[])
        self.add_state("coverage_fraction", default=[])

    def update(self, model, inputs, preds, target):
        if self.use_ground_truth:
            target_idx = target
        else:
            target_idx = torch.argmax(preds)
        # contribs = model.weight.data[target_idx] * inputs
        contribs = model.classifier.weight.data[target_idx] * inputs
        assert len(contribs.shape) == 1
        if self.use_only_positive:
            contribs = contribs.clamp(min=0)
        else:
            contribs = contribs.abs()
        contribs = contribs.sort(descending=True).values
        contribs_cumsum = contribs.cumsum(dim=0)
        sufficient_position = torch.where(
            contribs_cumsum >= self.energy_percent * contribs.sum())[0][0]
        self.coverage.append(sufficient_position.item())
        self.coverage_fraction.append(
            (sufficient_position / contribs.shape[0]).item())

    def compute(self):
        if len(self.coverage) == 0:
            return None
        return statistics.fmean(self.coverage)


def str2bool(v):
    return v.lower() in ('true')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which_machine", type=str, default="workstation",
                        choices=["workstation", "mlcloud", "laptop"])
    # parser.add_argument("--sae_dataset", type=str, default='wbcattr', choices=["cc3m", "wbcattr", "camelyon_panda"])
    parser.add_argument("--train_mode", type=str, default="train",
                        choices=["train", "eval"])

    parser.add_argument("--probe_dataset", type=str, default="camelyon",
                        choices=["wbcattr", "camelyon", "panda"])
    parser.add_argument("--feature_type", type=str, default='concept',
                        choices=["image", "concept"])
    # parser.add_argument("--perturbed_neuron", action="store_false", default=False)
    parser.add_argument('--perturbed_neuron', type=str2bool, default=False)
    parser.add_argument("--which_classifier", type=str, default='proto_mil',
                        choices=["linear_probe", "linear_probe_mil", "abmil",
                                 "proto_mil"])

    parser.add_argument("--probe_sparsity_loss", type=str, default="L1",
                        choices=["L1"])
    parser.add_argument("--probe_sparsity_loss_lambda", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size, use 1 for WSI datasets, and 64 for WBCATTR dataset")

    parser.add_argument("--probe_lr", type=float, default=1e-4)
    parser.add_argument("--autoencoder_input_dim", type=int, default=512,
                        help="input dimension of the autoencoder, we use conch, input dimension is 512")
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes in the dataset, wbcattr is 5, camelyon and panda is 2")

    parser.add_argument("--probe_val_freq", type=int, default=1)
    parser.add_argument("--probe_epochs", type=int, default=1,
                        help="number of epochs to train the linear probe, default 200")
    parser.add_argument("--probe_classification_loss",
                        type=str, default="CE", choices=["CE", "BCE"])
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument("--use_wandb", action="store_true", default=True)

    return parser


def update_args(args):
    if args.probe_dataset == 'wbcattr':
        args.num_classes = 5
    args.probe_nclasses = args.num_classes
    data_dirs = data_dir_dict[args.which_machine]
    args.log_dir = data_dirs["log_dir"]

    if args.train_mode == "train":
        if args.perturbed_neuron:
            perturbed_neuron = "with_perturbed_neuron_customized"
            exp_name = f"{args.which_classifier}_{args.probe_dataset}_feat:{args.feature_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}_lr{args.probe_lr}_epo{args.probe_epochs}_{perturbed_neuron}"
        else:
            exp_name = f"{args.which_classifier}_{args.probe_dataset}_feat:{args.feature_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}_lr{args.probe_lr}_epo{args.probe_epochs}"
        args.exp_name = exp_name

    if args.train_mode == "eval":
        eavl_exp_list = [
            "linear_probe_mil_panda_feat:concept_2024-12-10_18:01_lr0.0001_epo200",
            "linear_probe_mil_camelyon_feat:concept_2024-12-10_18:01_lr0.0001_epo200",
        ]
        search_key = args.which_classifier + "_" + args.probe_dataset + "_feat:" + args.feature_type
        for exp_path in eavl_exp_list:
            if search_key in exp_path:
                args.exp_name = exp_path
                args.model_path = os.path.join(data_dirs["exp_dir"], exp_path,
                                               "last_model.pth")

    args.exp_dir = os.path.join(data_dirs["exp_dir"], args.exp_name)
    # args.exp_dir = data_dirs["exp_dir"]
    args.src_folder = data_dirs["probe_dataset"][args.probe_dataset][
        "src_folder"]

    args.activation_dir = data_dirs["probe_dataset"][args.probe_dataset][
        "activation_dir"]
    args.concept_embeddings_dir = \
    data_dirs["probe_dataset"][args.probe_dataset]["concept_embeddings"]

    return args


def init_seed(manual_seed):
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    cudnn.benchmark = True


def init_experiment(args):
    init_seed(args.manual_seed)
    args = update_args(args)
    for dir in [args.log_dir, args.exp_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    init_wandb(args)


def init_wandb(args):
    if args.use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "3000"
        wandb._service_wait = 3000
        wandb.login(
            key='your_wandb_key')  # Set your wandb key
        wandb_project_name = "train_linear_probe"
        wandb_dir = os.path.join(args.log_dir, ".cache/")
        os.makedirs(wandb_dir, exist_ok=True)

        wandb.init(
            project=wandb_project_name,
            name=args.exp_name,
            dir=wandb_dir,
            config=args,
        )
        config = wandb.config
        config.lr = args.probe_lr
        config.probe_dataset = args.probe_dataset
        config.expansion_factor = args.expansion_factor
        config.which_classifier = args.which_classifier
        wandb.run.save()


def prepare_model(args, device):
    if args.feature_type == 'concept':
        input_dims = args.autoencoder_input_dim * args.expansion_factor
    if args.feature_type == 'image':
        input_dims = args.autoencoder_input_dim

    if args.probe_dataset in ["camelyon", "panda"]:
        assert args.which_classifier in ["abmil", "proto_mil"]

    if args.which_classifier == 'abmil':
        model = DefaultMILGraph(input_dims=input_dims,
                                output_dims=args.num_classes)
    if args.which_classifier == 'proto_mil':
        model = ProtoMIL(input_dims=input_dims, output_dims=args.num_classes)

    model = model.to(device)
    if args.train_mode == "eval":
        model_path = args.model_path
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.probe_lr)
    classification_loss_fn = torch.nn.CrossEntropyLoss()

    sparsity_loss_fn = None
    if args.probe_sparsity_loss is not None and args.which_classifier != 'abmil':
        sparsity_loss_fn = torch.nn.L1Loss()

    return model, optimizer, classification_loss_fn, sparsity_loss_fn


def init_dataloaders(args):
    if args.probe_dataset == 'camelyon' or args.probe_dataset == 'panda':

        if args.feature_type == 'concept':
            data_source = args.concept_embeddings_dir
        if args.feature_type == 'image':
            data_source = args.activation_dir

        train_df = pd.read_csv(os.path.join(args.src_folder, "train.csv"))
        valid_df = pd.read_csv(os.path.join(args.src_folder, "valid.csv"))
        test_df = pd.read_csv(os.path.join(args.src_folder, "test.csv"))

        if args.perturbed_neuron:
            assert args.feature_type == 'concept'
            if args.probe_dataset == 'camelyon':
                perturbed_neuron_list = [267, 439, 585, 908, 1509,
                                         1693]  ## adapted to camelyon
                # perturbed_neuron_list = [267, 439, 585, 908, 1509] # standard perturbed neurons
                # perturbed_neuron_list = [267, 1070, 1139, 1238] before

            if args.probe_dataset == 'panda':
                perturbed_neuron_list = [267, 339, 439, 585, 908, 1509,
                                         1568]  ## adapted to panda
                # perturbed_neuron_list = [267, 439, 585, 908, 1509] # standard perturbed neurons
                ###perturbed_neuron_list = [267, 585, 1139, 1976] before
        else:
            perturbed_neuron_list = []

        train_dataset = WSIEmbeddingDataset(data_source=data_source,
                                            df=train_df,
                                            index_col="slide_id",
                                            target_col="label",
                                            perturbed_neuron=perturbed_neuron_list)

        val_dataset = WSIEmbeddingDataset(data_source=data_source,
                                          df=valid_df,
                                          index_col="slide_id",
                                          target_col="label",
                                          perturbed_neuron=perturbed_neuron_list)

        test_dataset = WSIEmbeddingDataset(data_source=data_source,
                                           df=test_df,
                                           index_col="slide_id",
                                           target_col="label",
                                           perturbed_neuron=perturbed_neuron_list)  ### we can test on with perturbed neurons and without perturbed neurons

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                      num_workers=1)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                    num_workers=1)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                     num_workers=1)

    return train_dataloader, test_dataloader, val_dataloader


def train_model(args, model, optimizer, train_loader, val_dataloader,
                test_dataloader, classification_loss_fn, sparsity_loss_fn,
                device):
    model.train()
    best_val_auc = 0
    for e in range(args.probe_epochs):
        total_loss = 0
        total_classification_loss = 0
        total_sparsity_loss = 0
        total_batches = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):
            train_X = data['feature']
            train_y = data['label']
            total_batches += 1

            if args.which_classifier == 'linear_probe':
                train_X = train_X.to(device).squeeze()
                train_y = train_y.to(device).squeeze()
                out = model(train_X)
            if args.which_classifier in ['abmil', 'linear_probe_mil',
                                         'proto_mil']:
                train_X = train_X.to(device)
                train_y = train_y.to(device)
                results_dict = model(train_X)
                out = results_dict['bag_logits']
            classification_loss = classification_loss_fn(out, train_y)
            if sparsity_loss_fn is not None:
                # sparsity_loss = sparsity_loss_fn(
                #     model.weight.flatten(), torch.zeros_like(model.weight.flatten()))
                sparsity_loss = sparsity_loss_fn(
                    model.classifier.weight.flatten(),
                    torch.zeros_like(model.classifier.weight.flatten()))
                total_sparsity_loss += sparsity_loss.item()
            else:
                sparsity_loss = 0
            total_classification_loss += classification_loss.item()
            loss = classification_loss + args.probe_sparsity_loss_lambda * sparsity_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / total_batches
        avg_class_loss = total_classification_loss / total_batches
        avg_sparse_loss = total_sparsity_loss / total_batches
        print(
            f"Epoch: {e + 1}, Training Loss: {avg_loss}, Classification Loss: {total_classification_loss / total_batches}, Sparsity Loss: {total_sparsity_loss / total_batches}")

        if args.use_wandb:
            wandb.log(
                {"Total Loss": avg_loss, "Classification Loss": avg_class_loss,
                 "Sparsity Loss": avg_sparse_loss}, step=e + 1)

        if (e + 1) % args.probe_val_freq == 0:  ## what does this probe_val_freq mean?
            # if (e+1) % args.probe_eval_coverage_freq == 0:  ## what does this probe_eval_coverage_freq mean?
            #     val_loss, val_class_loss, val_sparse_loss, val_acc_top1, val_acc_top5, coverage_energy_wise, zero_count_dict = eval_model(
            #         model, train_val_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, device, eval_coverage=True)
            # else:
            #     val_loss, val_class_loss, val_sparse_loss, val_acc_top1, val_acc_top5, coverage_energy_wise, zero_count_dict = eval_model(
            #         model, train_val_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, device)

            val_loss, val_class_loss, val_sparse_loss, val_acc_top1, val_acc_top5, coverage_energy_wise, zero_count_dict, auc_value, avg_score = eval_model(
                model, val_dataloader, args.num_classes,
                args.probe_classification_loss, args.probe_sparsity_loss_lambda,
                device)

            test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, test_zero_count_dict, test_auc_value, test_avg_score = eval_model(
                model, test_dataloader, args.num_classes,
                args.probe_classification_loss,
                args.probe_sparsity_loss_lambda, device)

            if auc_value is not None:
                if auc_value > best_val_auc:
                    best_val_auc = auc_value
                    torch.save(model.state_dict(),
                               os.path.join(args.exp_dir, f"best_model.pth"))

            print(
                f"Validation Loss: {val_loss}, Validation Classification Loss: {val_class_loss}, Validation Sparsity Loss: {val_sparse_loss}, Validation Accuracy Top1: {val_acc_top1}, Validation Accuracy Top5: {val_acc_top5}")
            print(
                f"Test Loss: {test_loss}, Test Classification Loss: {test_class_loss}, Test Sparsity Loss: {test_sparse_loss}, Test Accuracy Top1: {test_acc_top1}, Test Accuracy Top5: {test_acc_top5}")
            print(f"AUC: {auc_value}, Average Score: {avg_score}")
            print(
                f"test_AUC: {test_auc_value}, test_Average Score: {test_avg_score}")

            if args.use_wandb:
                wandb.log({"Validation Loss": val_loss,
                           "Validation Classification Loss": val_class_loss,
                           "Validation Sparsity Loss": val_sparse_loss,
                           "Validation accuracy top1": val_acc_top1,
                           "Validation accuracy top5": val_acc_top5},
                          step=e + 1)
                wandb.log({"Test Loss": test_loss,
                           "Test Classification Loss": test_class_loss,
                           "Test Sparsity Loss": test_sparse_loss,
                           "Test accuracy top1": test_acc_top1,
                           "Test accuracy top5": test_acc_top5}, step=e + 1)
                wandb.log({"AUC": auc_value, "Average Score": avg_score})
                wandb.log({"test_AUC": test_auc_value,
                           "test_Average Score": test_avg_score})

                # if args.use_wandb:
                #     wandb.log({"AUC": auc_value, "Average Score": avg_score, "Bag Score Classwise": bag_score_classwise})

        torch.save(model.state_dict(),
                   os.path.join(args.exp_dir, f"last_model.pth"))


def eval_model(model, loader, num_classes, classification_loss_name,
               sparsity_loss_lambda, device, eval_coverage=False):
    """
    Evaluates model accuracy and sparsity during and after training.
    """
    if args.which_classifier in ['abmil', 'linear_probe_mil', 'proto_mil']:
        valid_labels = []
        valid_predictions = []
        valid_probs = []

    with torch.no_grad():
        model.eval()
        accuracy_top1 = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro").to(device)
        if args.probe_nclasses >= 5:
            accuracy_top5 = torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, top_k=5, average="micro").to(device)
        coverage_energy_wise = {}
        coverage_energy_wise_energies = [0.9, 0.95, 0.99]
        for energy in coverage_energy_wise_energies:
            coverage_energy_wise[energy] = {}
            coverage_energy_wise[energy][
                "positive"] = ImageWiseCoverageEnergyPercent(
                energy_percent=energy, use_ground_truth=False,
                use_only_positive=True).to(device)
            coverage_energy_wise[energy][
                "absolute"] = ImageWiseCoverageEnergyPercent(
                energy_percent=energy, use_ground_truth=False,
                use_only_positive=False).to(device)
        total_loss = 0
        total_classification_loss = 0

        total_batches = 0
        if classification_loss_name == "CE":
            classification_loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        if args.which_classifier != 'abmil':
            sparsity_loss_fn = torch.nn.L1Loss()
            total_sparsity_loss = 0
        else:
            sparsity_loss_fn = None
        for batch_idx, data in enumerate(tqdm(loader)):
            test_X = data['feature']
            test_y = data['label']
            # test_X = test_X.to(model.weight.dtype).to(device)
            # test_X = test_X.to(model.classifier.weight.dtype).to(device)
            if args.which_classifier == 'linear_probe':
                test_X = test_X.to(device).squeeze()
                test_y = test_y.to(device).squeeze()
                out = model(test_X)
            if args.which_classifier in ['abmil', 'linear_probe_mil', 'proto_mil']:
                test_X = test_X.to(device)
                test_y = test_y.to(device)
                results_dict = model(test_X)
                out = results_dict['bag_logits']

                Y_prob = results_dict['Y_prob']
                bag_prediction = results_dict['Y_hat']
                valid_labels.append(test_y.squeeze().cpu().numpy())
                valid_predictions.append(bag_prediction.squeeze().cpu().numpy())
                valid_probs.append(Y_prob.squeeze().cpu().numpy())

            classification_loss = classification_loss_fn(out, test_y)

            # sparsity_loss = sparsity_loss_fn(
            #     model.weight.flatten(), torch.zeros_like(model.weight.flatten()))
            if sparsity_loss_fn is not None:
                sparsity_loss = sparsity_loss_fn(
                    model.classifier.weight.flatten(),
                    torch.zeros_like(model.classifier.weight.flatten()))
                total_sparsity_loss += sparsity_loss.item()
                loss = classification_loss + sparsity_loss_lambda * sparsity_loss
            else:
                loss = classification_loss
            total_classification_loss += classification_loss.item()
            total_loss += loss.item()
            accuracy_top1.update(out, test_y)
            if args.probe_nclasses >= 5:
                accuracy_top5.update(out, test_y)
            if eval_coverage:
                for img_idx in range(test_X.shape[0]):
                    img = test_X[img_idx]
                    target = test_y[img_idx]
                    for energy in coverage_energy_wise_energies:
                        coverage_energy_wise[energy]["positive"].update(
                            model, img, out[img_idx], target)
                        coverage_energy_wise[energy]["absolute"].update(
                            model, img, out[img_idx], target)
            total_batches += 1
        if args.which_classifier != 'abmil':
            zero_count_dict = {}
            # zero_count = torch.where(model.weight.data.flatten() == 0)[0].shape[0]
            zero_count = \
            torch.where(model.classifier.weight.data.flatten() == 0)[0].shape[0]
            print(f"Zero count: {zero_count}")
            zero_count_dict[0] = zero_count
            for tol in [1e-9, 1e-6, 1e-3, 1e-1]:
                # zero_count_tol = torch.where(
                #     torch.abs(model.weight.data.flatten()) < tol)[0].shape[0]
                zero_count_tol = torch.where(
                    torch.abs(model.classifier.weight.data.flatten()) < tol)[
                    0].shape[0]
                # print(
                #     f"Zero count for tol {tol}: {zero_count_tol}, out of {model.weight.data.flatten().shape[0]}")
                print(
                    f"Zero count for tol {tol}: {zero_count_tol}, out of {model.classifier.weight.data.flatten().shape[0]}")
                zero_count_dict[tol] = zero_count_tol
        else:
            zero_count_dict = None

        acc_top1 = accuracy_top1.compute()
        if args.probe_nclasses >= 5:
            acc_top5 = accuracy_top5.compute()
        else:
            acc_top5 = 0.0
        if eval_coverage:
            coverage_energy_wise_metric_vals = {}
            for energy in coverage_energy_wise_energies:
                coverage_energy_wise_metric_vals[energy] = {}
                coverage_energy_wise_metric_vals[energy]["positive"] = \
                coverage_energy_wise[energy]["positive"].compute(
                )
                coverage_energy_wise_metric_vals[energy]["absolute"] = \
                coverage_energy_wise[energy]["absolute"].compute(
                )
                print(
                    f"Coverage for energy {energy} positive: {coverage_energy_wise_metric_vals[energy]['positive']}")
                print(
                    f"Coverage for energy {energy} absolute: {coverage_energy_wise_metric_vals[energy]['absolute']}")
        avg_loss = total_loss / total_batches
        avg_class_loss = total_classification_loss / total_batches
        if sparsity_loss_fn is not None:
            avg_sparse_loss = total_sparsity_loss / total_batches
        else:
            avg_sparse_loss = None

        model.train()

    if args.which_classifier in ['abmil', 'linear_probe_mil', 'proto_mil']:
        valid_labels = np.array(valid_labels).astype(int)
        valid_predictions = np.array(valid_predictions)
        valid_probs = np.array(valid_probs)
        auc_value = roc_auc_score(valid_labels, valid_probs[:, 1])

        bag_score = 0
        bag_score_classwise = np.zeros(args.num_classes)
        bag_number_classwise = np.zeros(args.num_classes)

        for i in range(0, len(val_dataloader)):
            bag_score = np.array_equal(valid_labels[i],
                                       valid_predictions[i]) + bag_score
            bag_score_classwise[valid_labels[i]] += np.array_equal(
                valid_labels[i], valid_predictions[i])
            bag_number_classwise[valid_labels[i]] += 1

        avg_score = bag_score / len(val_dataloader)
        bag_score_classwise = bag_score_classwise / bag_number_classwise

        # print(f"AUC: {auc_value}, Average Score: {avg_score}, Bag Score Classwise: {bag_score_classwise}")
        #
        # if args.use_wandb:
        #     wandb.log({"AUC": auc_value, "Average Score": avg_score, "Bag Score Classwise": bag_score_classwise})
        #
        # if args.train_mode == 'eval':
        #     print(f"test_AUC: {auc_value}, test_Average Score: {avg_score}, test_Bag Score Classwise: {bag_score_classwise}")
        #     if args.use_wandb:
        #         wandb.log({"test_AUC": auc_value, "test_Average Score": avg_score, "test_Bag Score Classwise": bag_score_classwise})

        return avg_loss, avg_class_loss, avg_sparse_loss, acc_top1, acc_top5, None, zero_count_dict, auc_value, avg_score

    if eval_coverage:
        return avg_loss, avg_class_loss, avg_sparse_loss, acc_top1, acc_top5, coverage_energy_wise_metric_vals, zero_count_dict, None, None
    return avg_loss, avg_class_loss, avg_sparse_loss, acc_top1, acc_top5, None, zero_count_dict, None, None


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = get_parser()
    args = parser.parse_args()
    init_experiment(args)
    train_dataloader, test_dataloader, val_dataloader = init_dataloaders(args)
    model, optimizer, classification_loss_fn, sparsity_loss_fn = prepare_model(
        args, device)

    print(model)
    if args.train_mode == 'train':
        train_model(args, model, optimizer, train_dataloader, val_dataloader,
                    test_dataloader, classification_loss_fn, sparsity_loss_fn,
                    device)
    if args.train_mode == 'eval':
        eval_model(model, test_dataloader, args.num_classes,
                   args.probe_classification_loss,
                   args.probe_sparsity_loss_lambda, device)

