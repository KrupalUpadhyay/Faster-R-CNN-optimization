import optuna
from optuna.trial import TrialState
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


classes = ["_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

num_classes = len(classes)

# Soft-NMS Function (from your code)
def soft_nms(boxes, scores, iou_thr=0.7, sigma=0.5, thresh=0.001):
    boxes = boxes.clone()
    scores = scores.clone()
    keep = []
    while boxes.size(0) > 0:
        max_idx = torch.argmax(scores)
        max_box = boxes[max_idx]
        keep.append(max_idx.item())
        if boxes.size(0) == 1:
            break
        ious = box_iou(max_box.unsqueeze(0), boxes)[0]
        scores = scores * torch.exp(- (ious ** 2) / sigma)
        mask = scores > thresh
        boxes, scores = boxes[mask], scores[mask]
    return keep

#  Objective Function for Optuna
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    step_size = trial.suggest_int("step_size", 1, 5)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)

    # Soft-NMS parameters
    iou_thr = trial.suggest_float("iou_thr", 0.3, 0.9)
    sigma = trial.suggest_float("sigma", 0.1, 1.0)
    thresh = trial.suggest_float("thresh", 0.0001, 0.01, log=True)

    # Build model
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
    model = FasterRCNN(backbone, num_classes=num_classes, box_nms_thresh=iou_thr)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer + Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training Loop (1–3 epochs for speed)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}", leave=False):
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(l for l in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

    # Evaluation using mAP
    metric = MeanAveragePrecision()
    model.eval()
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [i.to(device) for i in imgs]
            preds = model(imgs)

            # Apply Soft-NMS manually with trial parameters
            for pred in preds:
                keep_idx = soft_nms(pred["boxes"].cpu(), pred["scores"].cpu(),
                                    iou_thr=iou_thr, sigma=sigma, thresh=thresh)
                for k in pred.keys():
                    pred[k] = pred[k][keep_idx]

            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds, targets)

    results = metric.compute()
    map50 = results["map_50"].item() if "map_50" in results else 0.0

    # Report mAP to Optuna
    trial.report(map50, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return map50

# Run the Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=7200)  # 30 trials, 2 hours

#  Best Parameters & Saving
print("\n Best trial:")
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

torch.save(best_trial.params, "best_hyperparams_optuna_softnms.pth")
print("Best parameters saved to best_hyperparams_optuna_softnms.pth")

