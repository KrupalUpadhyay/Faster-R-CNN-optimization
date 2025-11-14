# ===============================================================
# Faster R-CNN with UN-SHARED VGG16 Backbone (Conv1–4 + Conv5)
# ===============================================================

import os
import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn
import random

# ===============================================================
# 1️⃣ Dataset Setup
# ===============================================================

train_img_dir = "/scratch/data/gehlot5/CV_Project_13-11-25/4/images/train/"
val_img_dir   = "/scratch/data/gehlot5/CV_Project_13-11-25/4/images/val/"
train_label_dir = "/scratch/data/gehlot5/CV_Project_13-11-25/4/labels/train/"
val_label_dir   = "/scratch/data/gehlot5/CV_Project_13-11-25/4/labels/val/"

classes = ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ===============================================================
# 2️⃣ Custom Dataset Loader
# ===============================================================

class CustomDataset(Dataset):
    def __init__(self, img_folder, label_folder, classes=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg','.jpeg','.png'))]
        self.classes = classes

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, os.path.splitext(img_name)[0] + '.txt')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, x_center, y_center, w, h = map(float, line.strip().split())
                    cls = int(cls) + 1

                    h_img, w_img = img.shape[1:]
                    x_min = (x_center - w/2) * w_img
                    y_min = (y_center - h/2) * h_img
                    x_max = (x_center + w/2) * w_img
                    y_max = (y_center + h/2) * h_img

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(cls)

        if len(boxes) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target

    def __len__(self):
        return len(self.img_files)


# ===============================================================
# 3️⃣ Load Datasets
# ===============================================================

train_dataset = CustomDataset(train_img_dir, train_label_dir, classes)
val_dataset   = CustomDataset(val_img_dir, val_label_dir, classes)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# ===============================================================
# 4️⃣ UN-SHARED VGG16 BACKBONE (Corrected)
# ===============================================================

class TwoBackbone(nn.Module):
    """
    Custom Backbone:
    - conv1–4 used for RPN processing
    - conv5 used for ROI extraction
    """
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16_bn(weights='IMAGENET1K_V1')

        self.rpn_layers = nn.Sequential(*list(vgg.features.children())[:33])   # conv1–4
        self.roi_layers = nn.Sequential(*list(vgg.features.children())[33:43]) # conv5

        self.out_channels = 512  # REQUIRED by Faster R-CNN

    def forward(self, x):
        x = self.rpn_layers(x)
        x = self.roi_layers(x)
        return {"roi": x}  # must match featmap_names=["roi"]


# ===============================================================
# 5️⃣ Faster R-CNN Model Builder (Fully Fixed)
# ===============================================================

def get_model(num_classes):

    backbone = TwoBackbone()

    # FIX: Custom Anchor Generator (ONE feature map only)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),    # tuple inside tuple
        aspect_ratios=((0.5, 1.0, 2.0),)     # tuple inside tuple
    )

    roi_pool = MultiScaleRoIAlign(
        featmap_names=["roi"],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool
    )

    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ===============================================================
# 6️⃣ Training
# ===============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(len(classes))
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Average Loss = {total_loss/len(train_loader):.4f}")
    lr_scheduler.step()


# ===============================================================
# 7️⃣ Save Model
# ===============================================================

torch.save(model.state_dict(), "fasterrcnn_unshared_vgg16.pth")
print("Model saved successfully.")


# ===============================================================
# 8️⃣ Evaluation
# ===============================================================

model.eval()

def compute_iou(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.array([0.0])
    iou = box_iou(pred_boxes, gt_boxes)
    return iou.max(dim=1).values.cpu().numpy()

ious, precisions = [], []

with torch.no_grad():
    for images, targets in tqdm(val_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, out in enumerate(outputs):
            pb = out["boxes"]
            ps = out["scores"]
            gb = targets[i]["boxes"].to(device)

            keep = ps > 0.5
            pb = pb[keep]

            iou = compute_iou(pb, gb)
            ious.extend(iou)

            tp = np.sum(iou > 0.5)
            fp = len(iou) - tp
            fn = len(gb) - tp
            prec = tp / (tp + fp + fn + 1e-6)
            precisions.append(prec)

print("\nPerformance:")
print("Mean IoU:", np.mean(ious))
print("mAP@0.5:", np.mean(precisions))


# ===============================================================
# 9️⃣ Save Predictions
# ===============================================================

os.makedirs("predictions", exist_ok=True)
idxs = random.sample(range(len(val_dataset)), 5)

for idx in idxs:
    img, _ = val_dataset[idx]
    img_d = img.to(device)

    with torch.no_grad():
        pred = model([img_d])[0]

    img_np = (img * 255).permute(1,2,0).byte().numpy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = classes[label]
            cv2.rectangle(img_np, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img_np, f"{class_name}:{score:.2f}",
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imwrite(f"predictions/pred_{idx}.png", img_np)
    print(f"Saved predictions/pred_{idx}.png")
