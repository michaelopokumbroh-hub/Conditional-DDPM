
# Lab 6 - Conditional DDPM  |  Step 1: Dataset

import os, json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def labels_to_onehot(object_list, obj2idx):
    vec = torch.zeros(len(obj2idx), dtype=torch.float32)
    for obj in object_list:
        if obj in obj2idx:
            vec[obj2idx[obj]] = 1.0
    return vec


class ICLEVRDataset(Dataset):
    def __init__(self, root, json_path, obj_json_path, img_size=64):
        super().__init__()
        self.img_dir = os.path.join(root, "iclevr")

        with open(obj_json_path) as f:
            self.obj2idx = json.load(f)
        self.num_classes = len(self.obj2idx)

        with open(json_path) as f:
            raw = json.load(f)

        self.samples = []
        for fname, obj_list in raw.items():
            p = os.path.join(self.img_dir, fname)
            if os.path.isfile(p):
                self.samples.append((fname, obj_list))

        print(f"[ICLEVRDataset] Found {len(self.samples)} training samples.")

        # Same as reference notebook — simple (0.5,0.5,0.5) normalization
        # maps [0,1] -> [-1,1]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, obj_list = self.samples[idx]
        img   = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img   = self.transform(img)
        label = labels_to_onehot(obj_list, self.obj2idx)
        return img, label


class ICLEVRTestDataset(Dataset):
    def __init__(self, json_path, obj_json_path):
        super().__init__()
        with open(obj_json_path) as f:
            self.obj2idx = json.load(f)
        self.num_classes = len(self.obj2idx)
        with open(json_path) as f:
            self.conditions = json.load(f)
        print(f"[ICLEVRTestDataset] {len(self.conditions)} conditions "
              f"from {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        return labels_to_onehot(self.conditions[idx], self.obj2idx)


def get_train_loader(root, obj_json_path, img_size=64,
                     batch_size=32, num_workers=4, shuffle=True):
    dataset = ICLEVRDataset(
        root=root,
        json_path=os.path.join(root, "train.json"),
        obj_json_path=obj_json_path,
        img_size=img_size,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    ), dataset.num_classes


def get_test_loader(json_path, obj_json_path, batch_size=32, num_workers=0):
    dataset = ICLEVRTestDataset(json_path, obj_json_path)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers), dataset.num_classes
