data_dir_dict = {
    "sae_dataset": {
        "camelyon_panda": {
            "activation_dir": "/home/susu/datasets/camelyon_panda/20x/features_conch_v1_pt",
            "concept_embeddings": "/home/susu/datasets/camelyon_panda/20x/ef4_concept_embeddings",
        },
    },
    "probe_dataset": {
        "camelyon": {
            "src_folder": "/home/susu/datasets/camelyon16/split_csv",
            "activation_dir": "/home/susu/datasets/trident_out_camelyon/20x_256px_0px_overlap/features_conch_v1_pt",
            "data_h5_dir": "/home/susu/datasets/trident_out_camelyon/20x_256px_0px_overlap/features_conch_v1",
            "data_slide_dir": "/media/susu/BA11-1CAC/camelyon16/WSI",
            "concept_embeddings":"/home/susu/datasets/camelyon16_20x/features/conch_v1/ef4_all_concept_embeddings",
        },
        "panda": {
            "src_folder": "/home/susu/datasets/panda_level0_256/split_csv",
            "activation_dir": "/home/susu/datasets/trident_out_panda/20x_256px_0px_overlap/features_conch_v1_pt",
            "data_h5_dir": "/home/susu/datasets/trident_out_panda/20x_256px_0px_overlap/features_conch_v1",
            "concept_embeddings": "/home/susu/datasets/panda_20x/features/conch_v1/ef4_all_concept_embeddings",
        },
    },
    "log_dir": "/home/susu/sae_exps/logs",
    "exp_dir": "/home/susu/sae_exps",
    "save_dir_root": '/home/susu/sae_exps',


}