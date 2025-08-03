import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import YoloMalariaDataset
from src.model import ViTOMD
from src.utils import yolo_loss, mean_average_precision
from tqdm import tqdm

def train():
    # Configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    img_size = 640
    batch_size = 8
    lr = 1e-4
    num_classes = 16
    val_split = 0.2

    # Model
    model = ViTOMD(num_classes=num_classes).to(device)

    # Dataset (now only one folder for images and labels)
    full_dataset = YoloMalariaDataset(
        img_dir="ViTO_MD/datasets/images",
        label_dir="ViTO_MD/datasets/labels",
        img_size=img_size,
        augment=True
    )

    # Split into train/val
    val_len = int(len(full_dataset) * val_split)
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)

            loss = yolo_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        map_score = mean_average_precision(model, val_loader, device)
        print(f"Validation mAP: {map_score:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"ViTO_MD/outputs/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
