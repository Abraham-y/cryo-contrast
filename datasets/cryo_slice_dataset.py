from torch.utils.data import Dataset
import numpy as np
import glob
import torchvision.transforms as T
import torch

class CryoSliceDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.files = sorted(glob.glob(f"{data_folder}/*.npy"))
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(128, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(3)], p=0.5),
            T.Normalize(mean=[0.0], std=[1.0])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]).astype(np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        img = np.expand_dims(img, axis=0)
        img1 = self.transform(torch.from_numpy(img))
        img2 = self.transform(torch.from_numpy(img))
        return img1, img2
