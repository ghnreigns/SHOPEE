import albumentations
from config import CONFIG

transforms_train = albumentations.Compose(
    [
        albumentations.Resize(
            CONFIG["TRAINING"]["IMAGE_SIZE"], CONFIG["TRAINING"]["IMAGE_SIZE"]
        ),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(
            p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)
        ),
        albumentations.HueSaturationValue(
            p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2
        ),
        albumentations.ShiftScaleRotate(
            p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20
        ),
        # albumentations.CoarseDropout(p=0.5),
        albumentations.Normalize(),
    ]
)

transforms_valid = albumentations.Compose(
    [
        albumentations.Resize(
            CONFIG["TRAINING"]["IMAGE_SIZE"], CONFIG["TRAINING"]["IMAGE_SIZE"]
        ),
        albumentations.Normalize(),
    ]
)
