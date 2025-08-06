# src/train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import YoloMalariaDataset
from model import MalariaNet
from utils import yolo_loss, mean_average_precision
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def train():
    # Configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    epochs = 100
    img_size = 640
    batch_size = 2  # Lowered to avoid OOM
    lr = 1e-4
    num_classes = 20  # Can be changed to Match your data.yaml
    val_split = 0.2

    # Ensure output directory exists
    os.makedirs("ViTO_MD/outputs", exist_ok=True)

    # Model
    model = MalariaNet(num_classes=num_classes).to(device)

    # Dataset
    full_dataset = YoloMalariaDataset(
        img_dir="datasets/images",
        label_dir="datasets/labels",
        img_size=img_size,
        augment=True
    )

    # Split dataset
    val_len = int(len(full_dataset) * val_split)
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()  # AMP scaler

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}")

        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)

            with autocast():  #  Mixed precision
                preds = model(imgs)
                loss = yolo_loss(preds, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        map_score = mean_average_precision(model, val_loader, device)
        print(f"Validation mAP: {map_score:.4f}")

        # Memory monitoring
        print(f"GPU Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # Save model
        torch.save(model.state_dict(), f"ViTO_MD/outputs/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
