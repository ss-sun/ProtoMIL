import os
from pathlib import Path
import torch
import numpy as np
import math
import datetime
import random
import wandb
from time import time
from data_dir_dict import data_dir_dict
import argparse
import os.path as osp

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def common_init(args, disable_make_dirs=False):
    autoencoder_input_dim_dict = {'conch_out': 512}

    set_seed(args.seed)
    args.config_name = f"{args.sae_dataset}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}_lr{args.lr}_l1coeff{args.l1_coeff}_ef{args.expansion_factor}_rf{args.resample_freq}_hook{args.hook_points[0]}_bs{args.train_sae_bs}_epo{args.num_epochs}"

    print(f"Config name: {args.config_name}")
    args.img_enc_name_for_saving = args.img_enc_name

    args.autoencoder_input_dim_dict = autoencoder_input_dim_dict
    args.ae_input_dim_dict_key = {}
    args.ae_input_dim_dict_key["img"] = f"{args.img_enc_name_for_saving}_{args.hook_points[0]}"
    args.save_dir = {}
    args.save_dir_sae_ckpts = {}
    args.save_dir["img"] = Path(osp.join(args.save_dir_root, f"{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.hook_points[0]}/{args.config_name}"))

    if not disable_make_dirs:
        os.makedirs(osp.join(args.save_dir_root,
                             f"{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.hook_points[0]}"), exist_ok=True)

    for modality in args.save_dir:
        if not disable_make_dirs:
            args.save_dir[modality].mkdir(exist_ok=True)
        args.save_dir_sae_ckpts[modality] = Path(
            osp.join(args.save_dir[modality], "sae_checkpoints"))

        if not disable_make_dirs:
            args.save_dir_sae_ckpts[modality].mkdir(exist_ok=True)

    args.enc_name = {}
    args.enc_name["img"] = args.img_enc_name


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dataset", type=str, default='camelyon_panda', choices=["camelyon_panda"])
    parser.add_argument("--img_enc_name", type=str, default='conch', help='Name of the clip image encoder', choices=["conch"])
    parser.add_argument("--modality", type=str, default="img", choices=["img"])
    parser.add_argument("--l1_coeff", type=float, default=3e-4)
    # Adam parameters (set to the default ones here)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument('--hook_points', nargs='*', help='Name of the model hook points to get the activations from', default=['out'])
    parser.add_argument("--resample_freq", type=int, default=500_000)
    parser.add_argument("--resample_dataset_size", type=int, default=819_200)
    parser.add_argument("--val_freq", type=int, default=50_000,help='number of samples after which to run validation')
    parser.add_argument("--ckpt_freq", type=int, default=250_000, help='number of samples after which to save the checkpoint')
    # SAE related
    parser.add_argument("--train_sae_bs", type=int, default=4096, help="batch size to train SAE")
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train the SAE")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--use_wandb", default=True)
    parser.add_argument("--wandb_entity", type=str, default="protomil")
    return parser




if __name__ == "__main__":
    import os
    from dncbm.custom_pipeline import Pipeline

    parser = get_common_parser()
    args = parser.parse_args()

    activation_dir = data_dir_dict["sae_dataset"][args.sae_dataset]["activation_dir"]
    save_dir_root = data_dir_dict["save_dir_root"]

    args.save_dir_root = save_dir_root
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_init(args)
    start_time = time()

    autoencoder_input_dim: int = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)

    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim,
                                    n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)

    loss = LossReducer(LearnedActivationsL1Loss(l1_coefficient=float(args.l1_coeff),), L2ReconstructionLoss(),)

    optimizer = AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
        lr=float(args.lr),
        betas=(float(args.adam_beta_1),
               float(args.adam_beta_2)),
        eps=float(args.adam_epsilon),
        weight_decay=float(args.adam_weight_decay),
        has_components_dim=True,
    )

    actual_resample_interval = 1
    activation_resampler = ActivationResampler(
        resample_interval=actual_resample_interval,
        n_activations_activity_collate=actual_resample_interval,
        max_n_resamples=math.inf,
        n_learned_features=n_learned_features, resample_epoch_freq=args.resample_freq,
        resample_dataset_size=args.resample_dataset_size,
    )

    if args.use_wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "3000"
        wandb._service_wait = 3000
        wandb.login(key='your_wandb_key')  # Set your wandb key
        wandb_project_name = "concept_discovery"
        wandb_dir = os.path.join(args.save_dir[args.modality], ".cache/")
        wandb_path = Path(wandb_dir)
        wandb_path.mkdir(exist_ok=True)
        wandb.init(
            project=wandb_project_name,
            dir=wandb_dir,
            name=args.config_name,
            config=args,)
        wandb.define_metric("custom_steps")
        wandb.define_metric("train/loss_instability_across_batches", step_metric="custom_steps")
        print("Wandb initialized")


    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(f"{args.save_dir_sae_ckpts[args.modality]}{args.save_suffix}"),
        loss=loss,
        optimizer=optimizer,
        device=args.device,
        args=args,
    )

    fnames = os.listdir(activation_dir)
    print(f"Getting fnames from {activation_dir}")

    train_fnames = []
    train_val_fnames = []

    random.shuffle(fnames)
    train_files = fnames[:int(0.8*len(fnames))]
    train_val_files = fnames[int(0.8*len(fnames)):]

    for fname in fnames:
        if fname.endswith('.pt') and fname in train_files:
            train_fnames.append(os.path.join(os.path.abspath(activation_dir), fname))
        elif fname.endswith('.pt') and fname in train_val_files:
            train_val_fnames.append(os.path.join(os.path.abspath(activation_dir), fname))

    print("len(train_fnames): ", len(train_fnames))
    print("len(train_val_fnames)", len(train_val_fnames))


    if args.val_freq == 0:
        train_fnames = train_fnames + train_val_fnames
        train_val_fnames = None

    print("Train and Train_val fnames created")

    pipeline.run_pipeline(
        train_batch_size=int(args.train_sae_bs),
        checkpoint_frequency=int(args.ckpt_freq),
        val_frequency=int(args.val_freq),
        num_epochs=args.num_epochs,
        train_fnames=train_fnames,
        train_val_fnames=train_val_fnames,
        start_time=start_time,
        resample_epoch_freq=args.resample_freq,
    )
    print(f"-------total time taken------ {np.round(time()-start_time,3)}")
