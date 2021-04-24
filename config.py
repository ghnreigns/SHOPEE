import numpy as np


CONFIG = {
    "COMPETITION_NAME": "Shopee - Price Match Guarantee",
    "MODEL": {"MODEL_NAME": "eca_nfnet_l0", "FC_DIM": 512, "DROPOUT": 0.2},
    "NUM_CLASSES": 11014,
    "TRAINING": {
        "IMAGE_SIZE": 256,
        "BATCH_SIZE": 8,
        "NUM_EPOCHS": 15,
        "USE_AMP": False,
        "NUM_WORKERS": 4,
        "ACCUMULATION_STEP": 1,
        "DEBUG": True,
        "DROP_LAST": True,
    },
    "VALIDATION": {"BATCH_SIZE": 16, "NUM_WORKERS": 4},
    "OPTIMIZER": {"Adam": {"lr": 1e-4}},
    "SCHEDULER": {
        "CosineAnnealingWarmRestarts": {"T_0": 15, "T_mult": 2, "eta_min": 0}
    },
    "ArcFace": {"scale": 64.0, "margin": 0.50, "easy_margin": False, "ls_eps": 0.0},
    "AUGMENTATION": {},
    "PATH": {
        "TRAINING_CSV": "/content/train.csv",
        "TRAIN_PATH": "/content/train_images",
        "SAVE_WEIGHT_PATH": "'/content/drive/My Drive/SHOPEE/WEIGHTS/VGG16",
    },
    "FOLD": 1,
    "SEARCH_SPACE": np.arange(40, 100, 10),
    "SEED": 1930,
}
