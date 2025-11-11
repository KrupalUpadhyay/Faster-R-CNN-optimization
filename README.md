
## Base Architecture: Faster R-CNN

**Faster R-CNN (Region-based Convolutional Neural Network)** is a **two-stage detector**:

1. The **Region Proposal Network (RPN)** first proposes candidate object regions.  
2. The **ROI Classifier and Box Regressor** then refine and classify each region.

---

### The Collaborative Attempt to Improve Accuracy and Experimentations
#### ** 1) Faster R-CNN (ResNet50 + FPN + Soft-NMS)**
- Used **ResNet50** as the backbone instead of VGG16 for deeper feature extraction.  
- Integrated **Feature Pyramid Network (FPN)** for multi-scale detection.  
- Added **Soft-NMS** to preserve overlapping detections.
#### ** 2) YOLO + SSD Hybrid**
- Combined **YOLO** (fast detection) with **SSD** (refinement on low-confidence regions).  
- Theoretically aimed for real-time detection with higher accuracy. 
