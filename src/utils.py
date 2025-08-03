# src/utils.py

import torch
import torch.nn as nn
import torchvision.ops as ops

def yolo_loss(preds, targets, lambda_coord=5, lambda_noobj=0.5):
    # Dummy loss function (replace with YOLO logic later)
    loss_fn = nn.MSELoss()
    return loss_fn(preds, torch.zeros_like(preds))  # placeholder

def mean_average_precision(model, loader, device):
    # Dummy mAP function
    return 0.5  # Replace with real mAP calculation later

def non_max_suppression(preds, conf_thresh=0.5, iou_thresh=0.5):
    return ops.nms(preds[:, :4], preds[:, 4], iou_thresh)
