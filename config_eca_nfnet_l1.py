import numpy as np


CONFIG = {
    "ID": "eca_nfnet_l1",
    "COMPETITION_NAME": "Shopee - Price Match Guarantee",
    "MODEL": {"MODEL_NAME": "eca_nfnet_l1", "FC_DIM": 512, "DROPOUT": 0.2},
    "NUM_CLASSES": 11014,
    "TRAINING": {
        "IMAGE_SIZE": 512,
        "BATCH_SIZE": 16,
        "NUM_EPOCHS": 16,
        "USE_AMP": True,
        "NUM_WORKERS": 4,
        "ACCUMULATION_STEP": 1,
        "DEBUG": False,
        "DROP_LAST": True,
    },
    "VALIDATION": {"BATCH_SIZE": 16, "NUM_WORKERS": 4},
    "OPTIMIZER": {"Adam": {"lr": 3e-5}},
    "SCHEDULER": {
        "CosineAnnealingLR": {"T_max": 16, "eta_min": 1e-7, "last_epoch": -1}
        # "CosineAnnealingWarmRestarts": {"T_0": 12, "T_mult": 2, "eta_min": 0}
    },
    "ArcFace": {"scale": 30, "margin": 0.5, "easy_margin": False, "ls_eps": 0.0},
    "AUGMENTATION": {},
    "PATH": {
        "TRAINING_CSV": "/content/train.csv",
        "TRAIN_PATH": "/content/train_images",
        "SAVE_WEIGHT_PATH": "/content/drive/My Drive/Shopee - Price Match Guarantee/WEIGHTS/eca_nfnet_l1/1st_MAY_V1/"
        # "SAVE_WEIGHT_PATH": "/content/drive/My Drive/Shopee - Price Match Guarantee/WEIGHTS/eca_nfnet_l0/30_APRIL_2021_V2/",
    },
    "FOLD": "ALL",
    "SEARCH_SPACE": np.arange(40, 100, 10),
    "SEED": 1930,
}
