# src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def compute_iou(self, box1, box2):
        """
        box = [x, y, w, h]
        """
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2

        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
        return iou

    def forward(self, preds, targets):
        """
        preds: [B, anchors, grid_size, grid_size, 5 + num_classes]
        targets: GT labels from dataset [B, num_objects, 5]
        """

        # For now, just a placeholder loss: total loss = sum of components
        # You'll plug this into the trainer later
        pred_boxes = preds[..., 0:4]
        pred_obj = preds[..., 4]
        pred_cls = preds[..., 5:]

        # Dummy targets for example
        # You can improve by using target assignment (anchor-matching, etc.)
        loss_box = self.mse(pred_boxes, torch.zeros_like(pred_boxes))
        loss_obj = self.bce(pred_obj, torch.zeros_like(pred_obj))
        loss_cls = self.bce(pred_cls, torch.zeros_like(pred_cls))

        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )

        return total_loss, loss_box, loss_obj, loss_cls
