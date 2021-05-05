import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from config_swin_small_patch4_window7_224 import CONFIG

# transforms_train = albumentations.Compose(
#     [
#         albumentations.Resize(
#             CONFIG["TRAINING"]["IMAGE_SIZE"], CONFIG["TRAINING"]["IMAGE_SIZE"]
#         ),
#         albumentations.HorizontalFlip(p=0.5),
#         albumentations.RandomBrightnessContrast(
#             p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)
#         ),
#         albumentations.HueSaturationValue(
#             p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2
#         ),
#         albumentations.ShiftScaleRotate(
#             p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20
#         ),
#         # albumentations.CoarseDropout(p=0.5),
#         albumentations.Normalize(),
#         ToTensorV2(p=1.0),
#     ]
# )

transforms_valid = albumentations.Compose(
    [
        albumentations.Resize(
            CONFIG["TRAINING"]["IMAGE_SIZE"], CONFIG["TRAINING"]["IMAGE_SIZE"]
        ),
        albumentations.Normalize(),
        ToTensorV2(p=1.0),
    ]
)

transforms_train = albumentations.Compose(
    [
        albumentations.Resize(
            CONFIG["TRAINING"]["IMAGE_SIZE"], CONFIG["TRAINING"]["IMAGE_SIZE"]
        ),
        albumentations.OneOf(
            [
                albumentations.GaussNoise(mean=15),
                albumentations.MotionBlur(p=0.2),
            ]
        ),
        albumentations.OneOf(
            [
                albumentations.RGBShift(
                    p=1.0,
                    r_shift_limit=(-10, 10),
                    g_shift_limit=(-10, 10),
                    b_shift_limit=(-10, 10),
                ),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.1, p=1
                ),
                albumentations.HueSaturationValue(hue_shift_limit=20, p=1),
            ],
            p=0.6,
        ),
        albumentations.OneOf(
            [
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
            ]
        ),
        albumentations.OneOf(
            [
                albumentations.IAAPerspective(p=0.3),
                albumentations.ElasticTransform(p=0.1),
            ]
        ),
        albumentations.OneOf(
            [
                albumentations.Rotate(limit=25, p=0.6),
                albumentations.IAAAffine(
                    scale=0.9,
                    translate_px=15,
                    rotate=25,
                    shear=0.2,
                ),
            ],
            p=1,
        ),
        albumentations.Cutout(
            num_holes=1,
            max_h_size=CONFIG["TRAINING"]["IMAGE_SIZE"] // 5,
            max_w_size=CONFIG["TRAINING"]["IMAGE_SIZE"] // 5,
            p=0.2,
        ),
        albumentations.Normalize(),
        ToTensorV2(p=1.0),
    ]
)
