# src/test.py

import os
import torch
import cv2
import numpy as np
from model import MalariaNet
from dataset import YoloMalariaDataset
from utils import non_max_suppression

# CONFIG
img_dir = "datasets/images"
label_dir = "datasets/labels"
model_path = "ViTO_MD/outputs/epoch_100.pt"  # ← or pick best epoch
img_size = 640
num_classes = 20  # match your dataset.yaml

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MalariaNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load sample images
image_files = sorted([
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

os.makedirs("ViTO_MD/test_outputs", exist_ok=True)

for filename in image_files[:10]:  # run on first 10 images
    img_path = os.path.join(img_dir, filename)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    # Postprocess
    boxes = []
    conf_thresh = 0.5
    iou_thresh = 0.4
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            row = output[y, x]
            conf = row[4]
            if conf > conf_thresh:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = int((x + cx - w / 2) * (img.shape[1] / output.shape[1]))
                y1 = int((y + cy - h / 2) * (img.shape[0] / output.shape[0]))
                x2 = int((x + cx + w / 2) * (img.shape[1] / output.shape[1]))
                y2 = int((y + cy + h / 2) * (img.shape[0] / output.shape[0]))
                cls = int(np.argmax(row[5:]))
                boxes.append([x1, y1, x2, y2, conf, cls])

    # Apply NMS
    if boxes:
        boxes = np.array(boxes)
        keep = non_max_suppression(torch.tensor(boxes), conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        boxes = boxes[keep.numpy()]

        # Draw boxes
        for (x1, y1, x2, y2, conf, cls) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"Class {int(cls)} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Save result
    out_path = f"ViTO_MD/test_outputs/{filename}"
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
