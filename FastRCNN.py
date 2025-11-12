# ===============================================
# 1️⃣ Install and Import Dependencies
# ===============================================
import os, torch, torchvision
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("✅ Device:", device)

# ===============================================
# 2️⃣ Dataset Paths & Classes
# ===============================================
train_img_dir  = "/scratch/data/m24cps008/CV_Project/4/images/train/"
val_img_dir    = "/scratch/data/m24cps008/CV_Project/4/images/val/"
train_label_dir= "/scratch/data/m24cps008/CV_Project/4/labels/train/"
val_label_dir  = "/scratch/data/m24cps008/CV_Project/4/labels/val"

classes = ["_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
num_classes = len(classes)

# ===============================================
# 3️⃣ Dataset Loader (YOLO-format → Pascal-style)
# ===============================================
class VOCDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt'))
        img = Image.open(img_path).convert("RGB")
        boxes, labels = [], []
        W, H = img.size
        with open(label_path) as f:
            for line in f.readlines():
                c, x, y, w, h = map(float, line.strip().split())
                xmin = (x - w/2) * W
                xmax = (x + w/2) * W
                ymin = (y - h/2) * H
                ymax = (y + h/2) * H
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(c)+1)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.as_tensor(labels, dtype=torch.int64)}
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self): 
        return len(self.images)

# ===============================================
# 4️⃣ Transform and Dataloaders
# ===============================================
transform = T.Compose([T.ToTensor()])
train_ds = VOCDataset(train_img_dir, train_label_dir, transform)
val_ds   = VOCDataset(val_img_dir, val_label_dir, transform)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
print("📁 Train:", len(train_ds), " | Val:", len(val_ds))

# ===============================================
# 5️⃣ Build Faster R-CNN (ResNet50 + FPN)
# ===============================================
backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
model = FasterRCNN(backbone, num_classes=num_classes, box_nms_thresh=0.7)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# ===============================================
# 6️⃣ Optimizer
# ===============================================
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.009603, momentum=0.7273, weight_decay=0.0000011)
lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.3519)

# ===============================================
# 7️⃣ Soft-NMS Function
# ===============================================
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

# ===============================================
# 8️⃣ Training Loop
# ===============================================
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(l for l in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    lr_scheduler.step()
    print(f"🧠 Epoch {epoch+1} | Avg Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "enhanced_faster_rcnn2_softnms.pth")
print("✅ Model saved to enhanced_faster_rcnn_softnms.pth")

# ===============================================
# 9️⃣ Evaluation Phase (mAP, Precision, Recall)
# ===============================================
metric = MeanAveragePrecision()
model.eval()

for imgs, targets in tqdm(val_loader, desc="Evaluating Model Accuracy"):
    imgs = [i.to(device) for i in imgs]
    with torch.no_grad():
        preds = model(imgs)
    preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
    targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
    metric.update(preds, targets)

results = metric.compute()

# ===============================================
# 🔟 Display Model Accuracy
# ===============================================
print("\n📊 ===== MODEL ACCURACY REPORT =====")
print(f"➡  mAP (IoU=0.5): {results['map_50']:.4f}")
print(f"➡  mAP (IoU=0.5:0.95): {results['map']:.4f}")
print(f"➡  Precision: {results['map_per_class'].mean():.4f}")
print(f"➡  Recall: {results['mar_100']:.4f}")
print("====================================\n")

# Per-class Average Precision
if 'map_per_class' in results and results['map_per_class'].ndim > 0:
    print("🎯 Per-Class AP Scores:")
    for i, ap in enumerate(results['map_per_class']):
        if i < len(classes):
            print(f"   {classes[i]}: {ap:.4f}")
else:
    print("⚠️ Per-class AP unavailable (returned as scalar).")

# ===============================================================
# 🔍 Visualization (Fixed for VOCDataset naming + label display)
# ===============================================================
os.makedirs("predictions", exist_ok=True)
import random

# Use val_ds if val_dataset not defined
dataset = val_dataset if 'val_dataset' in locals() else val_ds
indices = random.sample(range(len(dataset)), min(5, len(dataset)))

for idx in indices:
    img, _ = dataset[idx]
    img = img.to(device)

    with torch.no_grad():
        prediction = model([img])

    img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    conf_threshold = 0.2  # visualize more detections

    print(f"\n🖼️ Predictions for image {idx}:")
    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score >= conf_threshold:
            x_min, y_min, x_max, y_max = [int(i) for i in box.tolist()]
            class_name = classes[label.item()]
            print(f"  → {class_name} ({score:.2f}) at [{x_min}, {y_min}, {x_max}, {y_max}]")

            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_np, f"{class_name}: {score:.2f}", (x_min, max(15, y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    save_path = f"predictions/prediction_{idx}.png"
    cv2.imwrite(save_path, img_np)
    print(f"✅ Saved: {save_path}")

