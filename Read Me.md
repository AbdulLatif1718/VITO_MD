# 🧬 ViTO-MD: A Vision Transformer-Optimized Model for Malaria Parasite Detection

Welcome to **ViTO-MD**, a powerful malaria detection model built with:
- **YOLO dataset format**
- **TransformerLite blocks**
- **CBAM (Attention) modules**
- **Custom YOLO-style head**
- And inspired by **biological & mathematical principles** for intelligent disease recognition.

## 📂 Project Structure

ViTO_MD/
├── datasets/
│ ├── images/ # All malaria images (YOLO format)
│ └── labels/ # YOLO .txt labels (same name as images)
├── models/
│ └── vitomd.py # Lightweight YOLO-ResNet18 model
├── outputs/ # Saved checkpoints per epoch
├── src/
│ ├── dataset.py # Custom PyTorch Dataset class
│ ├── model.py # Final Transformer + CBAM + Conv model
│ ├── train.py # Training loop with val split
│ └── utils.py # Loss, mAP, NMS (to improve)
└── README.md




## 🚀 Setup (Colab or Jupyter)

# 1. Install dependencies
!pip install torch torchvision opencv-python --quiet
!pip install --upgrade numpy==1.24.4 --quiet

# Optional (if torchvision gives CUDA mismatch)
!pip uninstall -y torchvision
!pip install torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
🧠 Model Variants
Model Name	Description	Attention	Transformer	Head
ViTOMD	ResNet18 + YOLO-style	❌	❌	✅
MalariaNet	CBAM + TransformerLite	✅	✅	✅

🏋️ Training
Update this in src/train.py if needed:

img_dir = "ViTO_MD/datasets/images"
label_dir = "ViTO_MD/datasets/labels"
To train:


!python3 src/train.py
This will:
Split data 80% train / 20% val

Train for 100 epochs

Save model at: ViTO_MD/outputs/epoch_{n}.pt

🧪 Label Format (YOLOv5/8)
Ensure .txt label files are:


<class> <x_center> <y_center> <width> <height>
All values normalized (0–1), matching the image filename:
images/001.png
labels/001.txt
📊 Evaluation
mAP is calculated with a placeholder right now:


mean_average_precision(...) => returns 0.5
✅ You can integrate COCO mAP later using:

pycocotools

yolov5/utils/metrics.py

🧠 Mathematical Enhancements
This model includes:

CBAM: Convolutional Block Attention Module (channel + spatial)

TransformerLite: Lightweight self-attention over flattened feature maps

YOLO-style Loss: You can replace placeholder with true bounding box regression loss.

📝 To Improve
Replace placeholder loss (MSE) with YOLO loss (GIoU + obj + cls)

Add inference + visualization notebook

Convert to ONNX for edge deployment

🤝 Contributors
# Abdul Latif Sulley (@mrlogic)

# ViTO-MD Research Group

Guided by Human-AI interaction and deep mathematical concepts

📌 License
MIT — free to use and improve with credit.

