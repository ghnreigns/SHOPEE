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
from model_HN import *

# from config import CONFIG
# from config_efficientnet import CONFIG
# from config_seresnext50_32x4d import CONFIG

# from config_eca_nfnet_l1 import CONFIG
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
    from config_swin_small_patch4_window7_224 import CONFIG

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

    # model = HN_ARCFACE(
    #     num_classes=CONFIG["NUM_CLASSES"],
    #     dropout=CONFIG["MODEL"]["DROPOUT"],
    #     embedding_size=CONFIG["MODEL"]["FC_DIM"],
    #     backbone=CONFIG["MODEL"]["MODEL_NAME"],
    #     pretrained=True,
    # )
    model = HN_ARCFACE_SWIN_TRANSFORMER(
        num_classes=CONFIG["NUM_CLASSES"],
        dropout=CONFIG["MODEL"]["DROPOUT"],
        embedding_size=CONFIG["MODEL"]["FC_DIM"],
        backbone=CONFIG["MODEL"]["MODEL_NAME"],
        pretrained=True,
    )
    model = model.to(device)
    df = makeFold()

    dataset_train = SHOPEEDataset(df, "train", transform=transforms_train)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=CONFIG["TRAINING"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["TRAINING"]["NUM_WORKERS"],
        drop_last=CONFIG["TRAINING"]["DROP_LAST"],  # NEEDED FOR BN LAYERS TO DROP LAST.
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **CONFIG["OPTIMIZER"]["Adam"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, **CONFIG["SCHEDULER"]["CosineAnnealingLR"]
    )
    # This is built upon self.scheduler, note the params in self.schedule must match number of epochs.
    warmup_epoch = 1
    warmup_factor = 10
    from scheduler import GradualWarmupSchedulerV2

    # use initial lr divide by warmup factor
    scheduler_warmup = GradualWarmupSchedulerV2(
        optimizer,
        multiplier=10,
        total_epoch=warmup_epoch,
        after_scheduler=scheduler,
    )

    BEST_F1_SCORE = 0
    BEST_VAL_LOSS = 1000000
    BEST_TRAIN_LOSS = 100000
    for epoch in range(CONFIG["TRAINING"]["NUM_EPOCHS"]):
        scheduler_warmup.step(epoch)

        loss_train = train_func(
            model, train_loader
        )  # TODO: note this loss_train is the average of all losses, thus what you see inside the end is not a representative of the final averaged loss. As sin did not print it out.
        print("EPOCH | {} : LOSS | {}".format(epoch + 1, loss_train))
        run["TRAINING/LOSS_VALUE"].log(loss_train)

        if loss_train < BEST_TRAIN_LOSS:
            BEST_TRAIN_LOSS = loss_train
            torch.save(
                model.state_dict(),
                os.path.join(
                    CONFIG["PATH"]["SAVE_WEIGHT_PATH"],
                    "BEST_LOSS_EPOCH_{}_MODEL_{}_IMAGE_SIZE_{}.pt".format(
                        CONFIG["FOLD"],
                        CONFIG["MODEL"]["MODEL_NAME"],
                        str(CONFIG["TRAINING"]["IMAGE_SIZE"]),
                    ),
                ),
            )
