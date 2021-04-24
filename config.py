import numpy as np


CONFIG = {
    "COMPETITION_NAME": "Shopee - Price Match Guarantee",
    "MODEL": {"MODEL_NAME": "tf_efficientnet_b4_ns", "FC_DIM": 512, "DROPOUT": 0.2},
    "NUM_CLASSES": 11014,
    "TRAINING": {
        "IMAGE_SIZE": 512,
        "BATCH_SIZE": 8,
        "NUM_EPOCHS": 15,
        "USE_AMP": True,
        "NUM_WORKERS": 4,
        "ACCUMULATION_STEP": 1,
        "DEBUG": True,
        "DROP_LAST": True,
    },
    "VALIDATION": {"BATCH_SIZE": 8, "NUM_WORKERS": 4},
    "OPTIMIZER": {"Adam": {"lr": 1e-4}},
    "SCHEDULER": {
        "CosineAnnealingWarmRestarts": {"T_0": 15, "T_mult": 2, "eta_min": 0}
    },
    "ArcFace": {"scale": 10, "margin": 0.5, "easy_margin": False, "ls_eps": 0.0},
    "AUGMENTATION": {},
    "PATH": {
        "TRAINING_CSV": "/content/train.csv",
        "TRAIN_PATH": "/content/train_images",
        "SAVE_WEIGHT_PATH": "/content/drive/My Drive/SHOPEE/WEIGHTS/EfficientNet/B4",
    },
    "FOLD": 0,
    "SEARCH_SPACE": np.arange(40, 100, 10),
    "SEED": 1930,
}
