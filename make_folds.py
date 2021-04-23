import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
import os
from config import CONFIG


def makeFold():
    df_train = pd.read_csv(CONFIG["PATH"]["TRAINING_CSV"])
    df_train["file_path"] = df_train.image.apply(
        lambda x: os.path.join(CONFIG["PATH"]["TRAIN_PATH"], x)
    )

    gkf = GroupKFold(n_splits=5)
    df_train["fold"] = -1
    for fold, (train_idx, valid_idx) in enumerate(
        gkf.split(df_train, None, df_train.label_group)
    ):
        df_train.loc[valid_idx, "fold"] = fold

    label_encoder = LabelEncoder()
    df_train.label_group = label_encoder.fit_transform(df_train.label_group)
    return df_train
