import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import argparse

def load_model(model_path):
    
    print(f"[INFO] Loading model from: {model_path}")

    model = torch.load(model_path, map_location="cpu")

    model.eval()
    print("[INFO] Model loaded successfully.")
    return model


def preprocess_image(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb)

    return img, img_tensor


def draw_boxes(image, outputs, score_thresh=0.5):
   
    img = image.copy()
    boxes = outputs[0]["boxes"].detach().cpu().numpy()
    labels = outputs[0]["labels"].detach().cpu().numpy()
    scores = outputs[0]["scores"].detach().cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label}:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img


def main(args):
    # Load model
    model = load_model(args.model_path)

    # Load and preprocess image
    orig_img, img_tensor = preprocess_image(args.image_path)

    # Run inference
    with torch.no_grad():
        outputs = model([img_tensor])

    # Draw boxes
    result = draw_boxes(orig_img, outputs, score_thresh=args.score_thresh)

    # Save output
    save_path = "inference_output.jpg"
    cv2.imwrite(save_path, result)

    print(f"[INFO] Inference complete. Saved output to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model file (.pth)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Input image file")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                        help="Score threshold for displaying boxes")

    args = parser.parse_args()
    main(args)
