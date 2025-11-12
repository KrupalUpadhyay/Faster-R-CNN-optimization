# ===============================================================
# Hybrid YOLOv8 + SSD Detection Pipeline (Headless / MobaXterm)
# ===============================================================

import os, math
import torch, torchvision
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms as T
from ultralytics import YOLO

# ===============================================================
# 1️⃣  Setup
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

train_img_dir  = "/scratch/data/m24cps008/CV_Project/4/images/train/"
val_img_dir    = "/scratch/data/m24cps008/CV_Project/4/images/val"
train_label_dir= "/scratch/data/m24cps008/CV_Project/4/labels/train"
val_label_dir  = "/scratch/data/m24cps008/CV_Project/4/labels/val"

os.makedirs("results", exist_ok=True)

VOC_CLASSES = [
    "_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
num_classes = len(VOC_CLASSES)

# ===============================================================
# 2️⃣  Universal SSD Import (works on all Torchvision versions)
# ===============================================================
try:
    from torchvision.models.detection import ssd300_vgg, SSD300_VGG_Weights
    def load_ssd(pretrained=True, device="cpu"):
        weights = SSD300_VGG_Weights.COCO_V1 if pretrained else None
        model = ssd300_vgg(weights=weights)
        return model.to(device)
    print("📦 Using new API: torchvision.models.detection.ssd300_vgg")
except ImportError:
    from torchvision.models.detection.ssd import ssd300_vgg16, SSD300_VGG16_Weights
    def load_ssd(pretrained=True, device="cpu"):
        weights = SSD300_VGG16_Weights.COCO_V1 if pretrained else None
        model = ssd300_vgg16(weights=weights)
        return model.to(device)
    print("📦 Using legacy API: torchvision.models.detection.ssd300_vgg16")

# ===============================================================
# 3️⃣  Load YOLO + SSD models
# ===============================================================
yolo_model = YOLO("yolov8n.pt").to(device)
ssd_model  = load_ssd(pretrained=True, device=device)
ssd_model.eval()

ssd_transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ===============================================================
# 4️⃣  Helper functions
# ===============================================================
def bbox_iou_numpy(boxA, boxesB):
    xA = np.maximum(boxA[0,0], boxesB[:,0])
    yA = np.maximum(boxA[0,1], boxesB[:,1])
    xB = np.minimum(boxA[0,2], boxesB[:,2])
    yB = np.minimum(boxA[0,3], boxesB[:,3])
    interW = np.maximum(0.0, xB - xA)
    interH = np.maximum(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[0,2]-boxA[0,0]) * (boxA[0,3]-boxA[0,1])
    boxBArea = (boxesB[:,2]-boxesB[:,0]) * (boxesB[:,3]-boxesB[:,1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def soft_nms_numpy(boxes, scores, sigma=0.5, score_thresh=0.001):
    boxes, scores = boxes.copy(), scores.copy()
    N = boxes.shape[0]
    idxs = np.arange(N)
    keep = []
    while len(idxs) > 0:
        max_ind = np.argmax(scores[idxs])
        i = idxs[max_ind]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = bbox_iou_numpy(boxes[i:i+1], boxes[idxs])
        scores[idxs] = scores[idxs] * np.exp(-(ious**2)/sigma)
        idxs = idxs[scores[idxs] > score_thresh]
    return keep

def read_yolo_label(label_path, img_w, img_h):
    boxes, labels = [], []
    if not os.path.exists(label_path): 
        return np.zeros((0,4)), np.zeros((0,), dtype=np.int64)
    with open(label_path, "r") as f:
        for line in f.readlines():
            vals = line.strip().split()
            if len(vals) < 5:
                continue
            cls, cx, cy, w, h = map(float, vals[:5])
            xmin, ymin = (cx - w/2) * img_w, (cy - h/2) * img_h
            xmax, ymax = (cx + w/2) * img_w, (cy + h/2) * img_h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(cls) + 1)
    if len(boxes)==0:
        return np.zeros((0,4)), np.zeros((0,), dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

# ===============================================================
# 5️⃣  Parameters
# ===============================================================
YOLO_CONF_THRESH = 0.5
YOLO_LOW_THRESH  = 0.5
SSD_CONF_THRESH  = 0.3
SOFTNMS_SIGMA    = 0.5
SOFTNMS_THRESH   = 0.001

val_images = [f for f in os.listdir(val_img_dir) if f.endswith(".jpg")]
print(f"📸 Validation images found: {len(val_images)}")

metric = MeanAveragePrecision(class_metrics=True)

# ===============================================================
# 6️⃣  Hybrid Inference Loop
# ===============================================================
for img_name in tqdm(val_images, desc="Running YOLO+SSD Hybrid Inference"):
    img_path = os.path.join(val_img_dir, img_name)
    label_path = os.path.join(val_label_dir, img_name.replace(".jpg", ".txt"))
    pil_img = Image.open(img_path).convert("RGB")
    W, H = pil_img.size
    img_np = np.array(pil_img)

    # ---------- YOLO detection ----------
    results = yolo_model.predict(source=img_np, imgsz=640, conf=0.001, verbose=False)
    res = results[0]
    if hasattr(res, "boxes") and res.boxes is not None:
        yolo_xyxy = res.boxes.xyxy.cpu().numpy()
        yolo_conf = res.boxes.conf.cpu().numpy()
        yolo_cls  = res.boxes.cls.cpu().numpy().astype(int)
    else:
        yolo_xyxy = np.zeros((0,4))
        yolo_conf = np.zeros((0,))
        yolo_cls  = np.zeros((0,), dtype=int)

    keep_high = np.where(yolo_conf >= YOLO_CONF_THRESH)[0]
    keep_low  = np.where(yolo_conf < YOLO_LOW_THRESH)[0]

    final_boxes, final_scores, final_labels = [], [], []

    # Accept high-confidence YOLO boxes
    for i in keep_high:
        final_boxes.append(yolo_xyxy[i].tolist())
        final_scores.append(float(yolo_conf[i]))
        final_labels.append(int(yolo_cls[i]) + 1)

    # SSD refinement for low-confidence YOLO boxes
    for i in keep_low:
        bx = yolo_xyxy[i]
        xmin, ymin, xmax, ymax = map(int, [max(0,bx[0]), max(0,bx[1]), min(W,bx[2]), min(H,bx[3])])
        if xmax <= xmin or ymax <= ymin:
            continue
        crop = pil_img.crop((xmin, ymin, xmax, ymax))
        input_tensor = ssd_transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            ssd_out = ssd_model(input_tensor)[0]
        s_boxes = ssd_out["boxes"].cpu().numpy()
        s_scores = ssd_out["scores"].cpu().numpy()
        s_labels = ssd_out["labels"].cpu().numpy()
        if len(s_boxes) == 0:
            continue
        sx = (s_boxes[:,0] / 300.0) * (xmax - xmin) + xmin
        sy = (s_boxes[:,1] / 300.0) * (ymax - ymin) + ymin
        sx2 = (s_boxes[:,2] / 300.0) * (xmax - xmin) + xmin
        sy2 = (s_boxes[:,3] / 300.0) * (ymax - ymin) + ymin
        mapped_boxes = np.stack([sx, sy, sx2, sy2], axis=1)
        keep_idx = np.where(s_scores >= SSD_CONF_THRESH)[0]
        for k in keep_idx:
            final_boxes.append(mapped_boxes[k].tolist())
            final_scores.append(float(s_scores[k]))
            final_labels.append(int(s_labels[k]))

    # Apply Soft-NMS
    if len(final_boxes) > 0:
        boxes_arr = np.array(final_boxes, dtype=np.float32)
        scores_arr = np.array(final_scores, dtype=np.float32)
        labels_arr = np.array(final_labels, dtype=np.int64)
        keep_inds = soft_nms_numpy(boxes_arr, scores_arr, sigma=SOFTNMS_SIGMA, score_thresh=SOFTNMS_THRESH)
        boxes_arr, scores_arr, labels_arr = boxes_arr[keep_inds], scores_arr[keep_inds], labels_arr[keep_inds]
    else:
        boxes_arr = np.zeros((0,4), dtype=np.float32)
        scores_arr = np.zeros((0,), dtype=np.float32)
        labels_arr = np.zeros((0,), dtype=np.int64)

    preds = {"boxes": torch.tensor(boxes_arr, dtype=torch.float32),
             "scores": torch.tensor(scores_arr, dtype=torch.float32),
             "labels": torch.tensor(labels_arr, dtype=torch.int64)}

    gt_boxes, gt_labels = read_yolo_label(label_path, W, H)
    gt = {"boxes": torch.tensor(gt_boxes, dtype=torch.float32),
          "labels": torch.tensor(gt_labels, dtype=torch.int64)}

    metric.update([{k:v.cpu() for k,v in preds.items()}],
                  [{k:v.cpu() for k,v in gt.items()}])

    # ---------- Save image ----------
    img_cv = np.array(pil_img)[:,:,::-1].copy()
    for j, box in enumerate(boxes_arr):
        if scores_arr[j] >= YOLO_CONF_THRESH:
            x1, y1, x2, y2 = map(int, box)
            cls_name = VOC_CLASSES[labels_arr[j]] if labels_arr[j] < len(VOC_CLASSES) else str(labels_arr[j])
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_cv, f"{cls_name}:{scores_arr[j]:.2f}",
                        (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    save_path = f"results/{os.path.splitext(img_name)[0]}_det.png"
    cv2.imwrite(save_path, img_cv)

# ===============================================================
# 7️⃣  Evaluation
# ===============================================================
results = metric.compute()
print("\n===== HYBRID YOLO+SSD EVALUATION =====")
print(f"mAP (IoU=0.5): {results['map_50']:.4f}")
print(f"mAP (IoU=0.5:0.95): {results['map']:.4f}")
print(f"mAP per class (first 20): {results['map_per_class'][:20]}")
print("=====================================")
print(f"✅ All detection results saved to: {os.path.abspath('results/')}")

