import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image, ImageDraw

# Configuration
DATASET_ROOT = os.path.join(os.getcwd(), "VOCdevkit")  # Automatically detect
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "fasterrcnn_vgg16_voc.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Dataset root verified: {DATASET_ROOT}")

# Transforms
def get_transform():
    return T.Compose([T.ToTensor()])

# Dataset loading
train_dataset_07 = VOCDetection(
    root=DATASET_ROOT,
    year="2007",
    image_set="train",
    download=False,
    transforms=get_transform()
)

train_dataset_12 = VOCDetection(
    root=DATASET_ROOT,
    year="2012",
    image_set="train",
    download=False,
    transforms=get_transform()
)

train_dataset = ConcatDataset([train_dataset_07, train_dataset_12])

test_dataset = VOCDetection(
    root=DATASET_ROOT,
    year="2007",
    image_set="val",
    download=False,
    transforms=get_transform()
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model setup (Faster R-CNN with VGG16 backbone)
backbone = torchvision.models.vgg16(weights="IMAGENET1K_V1").features
backbone.out_channels = 512

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

model = FasterRCNN(
    backbone=backbone,
    num_classes=21,  # 20 classes + background
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

model.to(device)

# Training setup
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 5  # Adjust as needed

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for imgs, targets in pbar:
        imgs = [img.to(device) for img in imgs]
        formatted_targets = []

        for ann in targets:
            objs = ann["annotation"]["object"]
            if not isinstance(objs, list):
                objs = [objs]
            boxes = []
            labels = []
            for obj in objs:
                bbox = obj["bndbox"]
                boxes.append([
                    float(bbox["xmin"]),
                    float(bbox["ymin"]),
                    float(bbox["xmax"]),
                    float(bbox["ymax"])
                ])
                labels.append(1) 

            formatted_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                "labels": torch.tensor(labels, dtype=torch.int64, device=device)
            })

        loss_dict = model(imgs, formatted_targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        pbar.set_postfix(loss=losses.item())

    lr_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Evaluation
model.eval()
sample_imgs, sample_targets = next(iter(test_loader))
imgs = [img.to(device) for img in sample_imgs]

with torch.no_grad():
    preds = model(imgs)

def draw_boxes(image, boxes, labels, scores=None):
    img = T.ToPILImage()(image.cpu()).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        box = box.tolist()
        label = labels[i].item()
        color = "red"
        draw.rectangle(box, outline=color, width=2)
        text = f"Class: {label}"
        if scores is not None:
            text += f" ({scores[i]:.2f})"
        draw.text((box[0], box[1]), text, fill=color)
    return img

for i in range(min(5, len(preds))):
    boxes = preds[i]['boxes'].detach().cpu()
    labels = preds[i]['labels'].detach().cpu()
    scores = preds[i]['scores'].detach().cpu()
    img = draw_boxes(imgs[i], boxes, labels, scores)
    img.save(os.path.join(OUTPUT_DIR, f"prediction_{i}.jpg"))

print("Sample predictions saved in 'outputs/' directory.")

# Custom NMS Section
def apply_custom_nms(prediction, iou_thresh=0.4):
    keep = torchvision.ops.nms(prediction["boxes"], prediction["scores"], iou_thresh)
    prediction["boxes"] = prediction["boxes"][keep]
    prediction["labels"] = prediction["labels"][keep]
    prediction["scores"] = prediction["scores"][keep]
    return prediction

for i in range(min(5, len(preds))):
    nms_pred = apply_custom_nms(preds[i])
    img_nms = draw_boxes(imgs[i], nms_pred["boxes"], nms_pred["labels"], nms_pred["scores"])
    img_nms.save(os.path.join(OUTPUT_DIR, f"prediction_NMS_{i}.jpg"))

print("Custom NMS outputs saved.")
