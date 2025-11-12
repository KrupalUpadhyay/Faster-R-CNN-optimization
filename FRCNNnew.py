# ===============================================================
#   CV Project - Faster R-CNN on Pascal VOC (MobaXterm Version)
# ===============================================================

import os
import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

# ===============================================================
# 1️⃣  Setup and Dependencies
# ===============================================================
# Make sure to run this once before running script (if needed):
# pip install torch torchvision torchaudio matplotlib opencv-python tqdm kagglehub

import kagglehub
path = kagglehub.dataset_download("sergiomoy/pascal-voc-2007-for-yolo")
print("✅ Dataset downloaded to:", path)

# ===============================================================
# 2️⃣  Define Dataset Paths
# ===============================================================
<<<<<<< HEAD
train_img_dir = os.path.join(path, "/scratch/gb_cod2/CV_Project/4/images/train/")
val_img_dir = os.path.join(path, "/scratch/gb_cod2/CV_Project/4/images/val/")
train_label_dir = os.path.join(path, "/scratch/gb_cod2/CV_Project/4/labels/train/")
val_label_dir = os.path.join(path, "/scratch/gb_cod2/CV_Project/4/labels/val/")
=======
train_img_dir = os.path.join(path, "/scratch/gb_cod4/CV_Project/4/images/train/")
val_img_dir = os.path.join(path, "/scratch/gb_cod4/CV_Project/4/images/val/")
train_label_dir = os.path.join(path, "/scratch/gb_cod4/CV_Project/4/labels/train/")
val_label_dir = os.path.join(path, "/scratch/gb_cod4/CV_Project/4/labels/val/")
>>>>>>> 00eac40ca9c366ef0795bee6fab71f1865376e24

# ===============================================================
# 3️⃣  Custom Dataset Class
# ===============================================================
class CustomDataset(Dataset):
    def __init__(self, img_folder, label_folder, classes=None, transforms=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transforms = transforms
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.classes = classes

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, os.path.splitext(img_name)[0] + '.txt')

        img = cv2.imread(img_path)[:, :, ::-1].copy()  # BGR → RGB
        img = F.to_tensor(img)

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    try:
                        cls, x_center, y_center, w, h = map(float, line.strip().split())
                        cls = int(cls) + 1
                        h_img, w_img = img.shape[1:]
                        x_min = (x_center - w / 2) * w_img
                        y_min = (y_center - h / 2) * h_img
                        x_max = (x_center + w / 2) * w_img
                        y_max = (y_center + h / 2) * h_img
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(cls)
                    except ValueError:
                        print(f"⚠️ Skipping invalid line in {label_path}")

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_files)

# ===============================================================
# 4️⃣  Load Dataset
# ===============================================================
classes = ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

train_dataset = CustomDataset(train_img_dir, train_label_dir, classes)
val_dataset = CustomDataset(val_img_dir, val_label_dir, classes)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ===============================================================
# 5️⃣  Model Setup
# ===============================================================
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = len(classes)
model = get_model(num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ Using device:", device)

# ===============================================================
# 6️⃣  Training
# ===============================================================
<<<<<<< HEAD
optimizer = torch.optim.SGD(model.parameters(), lr=0.009631, momentum=0.727, weight_decay=0.0000011)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.351)
num_epochs = 200
=======
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 20
>>>>>>> 00eac40ca9c366ef0795bee6fab71f1865376e24

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images, targets)
        losses = sum(output.values()) if isinstance(output, dict) else sum(output)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_train_loss += losses.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"📉 Epoch [{epoch+1}/{num_epochs}] Avg Training Loss: {avg_train_loss:.4f}")
    lr_scheduler.step()

# ===============================================================
# 7️⃣  Save Model
# ===============================================================
torch.save(model.state_dict(), "fasterrcnn2_trained_model.pth")
print("💾 Model saved as fasterrcnn_trained_model.pth")

# ===============================================================
# 8️⃣  Accuracy Calculation (No Validation Loss)
# ===============================================================
model.eval()
model.load_state_dict(torch.load("fasterrcnn2_trained_model.pth", map_location=device))

def compute_iou(pred_boxes, true_boxes):
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return np.array([0.0])
    # Ensure both tensors are on the same device
    device = pred_boxes.device
    true_boxes = true_boxes.to(device)
    iou = box_iou(pred_boxes, true_boxes)
    return iou.max(dim=1).values.detach().cpu().numpy()


ious_all, precisions = [], []

with torch.no_grad():
    for images, targets in tqdm(val_loader, desc="Evaluating Accuracy"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            pred_boxes = output['boxes']
            pred_scores = output['scores']
            gt_boxes = targets[i]['boxes']

            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]

            ious = compute_iou(pred_boxes, gt_boxes)
            ious_all.extend(ious)

            tp = np.sum(ious > 0.5)
            fp = len(ious) - tp
            fn = len(gt_boxes) - tp
            precision = tp / (tp + fp + fn + 1e-6)
            precisions.append(precision)

mean_iou = np.mean(ious_all) if len(ious_all) else 0
mean_precision = np.mean(precisions) if len(precisions) else 0
mAP = mean_precision

print("\n✅ Final Accuracy Results:")
print(f"➡️ Mean IoU: {mean_iou:.4f}")
print(f"➡️ Mean Precision (mAP@0.5): {mAP:.4f}")

# ===============================================================
# 9️⃣  Save Predicted Images
# ===============================================================
os.makedirs("predictions", exist_ok=True)
import random

indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))

for idx in indices:
    img, _ = val_dataset[idx]
    img = img.to(device)
    with torch.no_grad():
        prediction = model([img])

    img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score > 0.5:
            x_min, y_min, x_max, y_max = [int(i) for i in box.tolist()]
            class_name = classes[label.item()]
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_np, f"{class_name}: {score:.2f}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    save_path = f"predictions/prediction_{idx}.png"
    cv2.imwrite(save_path, img_np)
    print(f"✅ Saved: {save_path}")

