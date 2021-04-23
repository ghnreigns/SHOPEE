import torch
import cv2
import numpy as np


class SHOPEEDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res["image"]

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)

        img = torch.tensor(img).float()
        label = torch.tensor(row.label_group).float()

        if self.mode == "test":
            return img
        else:
            return img, label