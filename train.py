import torch
import torch.nn as nn
import timm
import os
import numpy as np
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm

from loss import ArcModule
from dataset import SHOPEEDataset
from model import *
from config import CONFIG
from threshold import find_threshold


def train_func(model, train_loader):
    model.train()
    bar = tqdm(train_loader)
    if CONFIG["TRAINING"]["USE_AMP"]:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):

        images, targets = images.to(device), targets.to(device).long()

        if CONFIG["TRAINING"]["DEBUG"] and batch_idx == 100:
            print("Debug Mode. Only train on first 100 batches.")
            break

        if CONFIG["TRAINING"]["USE_AMP"]:
            with torch.cuda.amp.autocast():

                logits = model(images, targets)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            if ((batch_idx + 1) % CONFIG["TRAINING"]["ACCUMULATION_STEP"] == 0) or (
                (batch_idx + 1) == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images, targets)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f"loss: {loss.item():.5f}, smth: {smooth_loss:.5f}")

    loss_train = np.mean(losses)
    return loss_train


def valid_func(model, valid_loader):
    model.eval()
    bar = tqdm(valid_loader)

    PROB = []
    TARGETS = []
    losses = []
    PREDS = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):

            images, targets = images.to(device), targets.to(device).long()

            logits = model(images, targets)

            PREDS += [torch.argmax(logits, 1).detach().cpu()]
            TARGETS += [targets.detach().cpu()]

            loss = criterion(logits, targets)
            losses.append(loss.item())

            bar.set_description(f"loss: {loss.item():.5f}")

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    accuracy = (PREDS == TARGETS).mean()

    loss_valid = np.mean(losses)
    return loss_valid, accuracy


def generate_test_features(model, test_loader):
    model.eval()
    bar = tqdm(test_loader)

    FEAS = []
    TARGETS = []

    with torch.no_grad():
        for batch_idx, (images) in enumerate(bar):

            images = images.to(device)

            features = model(images)

            FEAS += [features.detach().cpu()]

    FEAS = torch.cat(FEAS).cpu().numpy()

    return FEAS


if __name__ == "__main__":
    from seed import seed_all
    from make_folds import makeFold
    from transforms import transforms_train, transforms_valid
    import neptune.new as neptune

    seed_all(CONFIG["SEED"])

    device = "cuda"
    run = neptune.init(
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMjBlZDIwYy05MjJjLTRiYzgtOGVlNi1hYWRiYWYxNGViMzIifQ==",
        project="reighns/KAGGLE-SHOPEE-Price-Match-Guarantee",
    )

    run["Params"] = CONFIG

    if not os.path.exists(CONFIG["PATH"]["SAVE_WEIGHT_PATH"]):
        print("new save folder created")
        os.makedirs(CONFIG["PATH"]["SAVE_WEIGHT_PATH"])

    # model = SHOPEE_EfficientNetB4(
    #     num_classes=CONFIG["NUM_CLASSES"],
    #     dropout=CONFIG["MODEL"]["DROPOUT"],
    #     embedding_size=CONFIG["MODEL"]["FC_DIM"],
    #     backbone=CONFIG["MODEL"]["MODEL_NAME"],
    #     pretrained=True,
    # )

    model = SHOPEE_PLEASE_HIRE_US(
        num_classes=CONFIG["NUM_CLASSES"],
        dropout=CONFIG["MODEL"]["DROPOUT"],
        embedding_size=CONFIG["MODEL"]["FC_DIM"],
        backbone=CONFIG["MODEL"]["MODEL_NAME"],
        pretrained=True,
    )

    model = model.to(device)
    df = makeFold()
    df_train = df[df["fold"] != CONFIG["FOLD"]]
    df_valid = df[df["fold"] == CONFIG["FOLD"]]

    df_valid["count"] = df_valid.label_group.map(
        df_valid.label_group.value_counts().to_dict()
    )

    dataset_train = SHOPEEDataset(df_train, "train", transform=transforms_train)
    dataset_valid = SHOPEEDataset(df_valid, "valid", transform=transforms_valid)
    dataset_test = SHOPEEDataset(df_valid, "test", transform=transforms_valid)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=CONFIG["TRAINING"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["TRAINING"]["NUM_WORKERS"],
        drop_last=CONFIG["TRAINING"]["DROP_LAST"],  # NEEDED FOR BN LAYERS TO DROP LAST.
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=CONFIG["VALIDATION"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["VALIDATION"]["NUM_WORKERS"],
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=CONFIG["VALIDATION"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["VALIDATION"]["NUM_WORKERS"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **CONFIG["OPTIMIZER"]["Adam"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, **CONFIG["SCHEDULER"]["CosineAnnealingWarmRestarts"]
    )
    BEST_F1_SCORE = 0
    BEST_VAL_LOSS = 1000000
    BEST_TRAIN_LOSS = 100000
    for epoch in range(CONFIG["TRAINING"]["NUM_EPOCHS"]):
        scheduler.step()
        loss_train = train_func(
            model, train_loader
        )  # TODO: note this loss_train is the average of all losses, thus what you see inside the end is not a representative of the final averaged loss. As sin did not print it out.
        loss_valid, valid_accuracy = valid_func(model, valid_loader)
        run["TRAINING/LOSS_VALUE"].log(loss_train)
        run["VALIDATION/LOSS_VALUE"].log(loss_valid)
        run["VALIDATION/ACCURACY"].log(valid_accuracy)

        if loss_train < BEST_TRAIN_LOSS:
            BEST_TRAIN_LOSS = loss_train
            torch.save(
                model.state_dict(),
                os.path.join(
                    CONFIG["PATH"]["SAVE_WEIGHT_PATH"],
                    "BEST_LOSS_FOLD_{}_EPOCH_{}_MODEL_{}_IMAGE_SIZE_{}.pt".format(
                        CONFIG["FOLD"],
                        epoch,
                        CONFIG["MODEL"]["MODEL_NAME"],
                        str(CONFIG["TRAINING"]["IMAGE_SIZE"]),
                    ),
                ),
            )

        if epoch % 2 == 0:
            print(
                "Now generating features for the validation set to simulate the submission."
            )
            FEAS = generate_test_features(model, test_loader)
            FEAS = torch.tensor(FEAS).cuda()
            print("Finding Best Threshold in the given search space.")
            best_score, best_threshold = find_threshold(
                features=FEAS,
                df=df_valid,
                lower_count_thresh=0,
                upper_count_thresh=999,
                search_space=CONFIG["SEARCH_SPACE"],
            )
            if best_score >= BEST_F1_SCORE:
                BEST_F1_SCORE = best_score
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        CONFIG["PATH"]["SAVE_WEIGHT_PATH"],
                        "BEST_F1_SCORE{}_EPOCH_{}_MODEL_{}_IMAGE_SIZE_{}.pt".format(
                            CONFIG["FOLD"],
                            epoch,
                            CONFIG["MODEL"]["MODEL_NAME"],
                            str(CONFIG["TRAINING"]["IMAGE_SIZE"]),
                        ),
                    ),
                )