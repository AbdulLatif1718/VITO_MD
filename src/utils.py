import torch
import torchvision
import numpy as np

def xywh_to_xyxy(boxes):
    """Convert YOLO box format [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def non_max_suppression(prediction, conf_thresh=0.5, iou_thresh=0.5):
    """Performs NMS on predictions from one image"""
    boxes = []
    scores = []

    for i in range(prediction.shape[0]):
        pred = prediction[i]
        obj_conf = pred[4]
        class_probs = pred[5:]

        class_id = torch.argmax(class_probs)
        score = obj_conf * class_probs[class_id]

        if score > conf_thresh:
            box = pred[:4]
            boxes.append(box)
            scores.append(score)

    if not boxes:
        return []

    boxes = torch.stack(boxes)
    scores = torch.tensor(scores)
    boxes = xywh_to_xyxy(boxes)

    keep = torchvision.ops.nms(boxes, scores, iou_thresh)
    final_preds = [(boxes[i], scores[i]) for i in keep]
    return final_preds

def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-6
    return inter_area / union_area

def mean_average_precision(model, loader, device, iou_threshold=0.5):
    """Compute mAP@0.5 for a batch of images"""
    model.eval()
    total_true_positives = 0
    total_predictions = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            preds = preds.cpu()

            B, _, H, W = preds.shape
            preds = preds.permute(0, 2, 3, 1).reshape(B, -1, preds.shape[1])  # (B, N, C+5)

            for b in range(B):
                pred_boxes = non_max_suppression(preds[b])

                # Skip if no targets at all
                if targets.ndim == 1:
                    continue

                target_per_image = targets[b]

                # Skip empty targets
                if target_per_image.ndim == 1 or target_per_image.shape[0] == 0:
                    continue

                # Get GT boxes (in xyxy)
                gt_boxes = target_per_image[:, 1:5] * H
                gt_boxes = xywh_to_xyxy(gt_boxes)

                matched = set()
                for pred_box, _ in pred_boxes:
                    for i, gt_box in enumerate(gt_boxes):
                        if i in matched:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > iou_threshold:
                            total_true_positives += 1
                            matched.add(i)
                            break

                total_predictions += len(pred_boxes)

    if total_predictions == 0:
        return 0.0

    precision = total_true_positives / total_predictions
    return precision
