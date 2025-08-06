# src/dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class YoloMalariaDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def load_labels(self, label_path):
        if not os.path.exists(label_path):
            return np.zeros((0, 5), dtype=np.float32)
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    labels.append(parts)
        return np.array(labels, dtype=np.float32)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].rsplit('.', 1)[0] + ".txt")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Resize and normalize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)

        # Load and process labels
        targets = self.load_labels(label_path)
        if targets.size > 0:
            targets[:, 1:] *= self.img_size  # Scale bbox to image size

        targets = torch.tensor(targets, dtype=torch.float32)

        return img, targets
