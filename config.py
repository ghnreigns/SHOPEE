import numpy as np


CONFIG = {
    "COMPETITION_NAME": "Shopee - Price Match Guarantee",
    "MODEL": {"MODEL_NAME": "eca_nfnet_l1", "FC_DIM": 512, "DROPOUT": 0.2},
    "NUM_CLASSES": 11014,
    "TRAINING": {
        "IMAGE_SIZE": 512,
        "BATCH_SIZE": 16,
        "NUM_EPOCHS": 25,
        "USE_AMP": True,
        "NUM_WORKERS": 4,
        "ACCUMULATION_STEP": 1,
        "DEBUG": False,
        "DROP_LAST": True,
    },
    "VALIDATION": {"BATCH_SIZE": 16, "NUM_WORKERS": 4},
    "OPTIMIZER": {"Adam": {"lr": 1e-4}},
    "SCHEDULER": {
        "CosineAnnealingWarmRestarts": {"T_0": 25, "T_mult": 2, "eta_min": 0}
    },
    "ArcFace": {"scale": 30, "margin": 0.5, "easy_margin": False, "ls_eps": 0.0},
    "AUGMENTATION": {},
    "PATH": {
        "TRAINING_CSV": "/content/train.csv",
        "TRAIN_PATH": "/content/train_images",
        "SAVE_WEIGHT_PATH": "/content/drive/My Drive/SHOPEE/WEIGHTS/eca_nfnet_l1/26_APRIL_2021/",
    },
    "FOLD": 0,
    "SEARCH_SPACE": np.arange(40, 100, 10),
    "SEED": 1930,
}
