# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import os

# Load pretrained model with updated weights
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

# Define animal classes only from COCO dataset
ANIMAL_CLASSES = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
]

# Define COCO classes as provided
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    
    # Lấy ra danh sách dự đoán nhãn, hộp và điểm số
    pred_labels = list(pred[0]['labels'].numpy())
    pred_boxes = list(pred[0]['boxes'].detach().numpy())
    pred_scores = list(pred[0]['scores'].detach().numpy())
    
    # Lọc dựa trên ngưỡng điểm số và chỉ giữ lại các lớp động vật
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = []
    
    for idx, score in enumerate(pred_scores):
        if score > threshold:
            class_name = COCO_INSTANCE_CATEGORY_NAMES[pred_labels[idx]]
            if class_name in ANIMAL_CLASSES:
                filtered_boxes.append([(pred_boxes[idx][0], pred_boxes[idx][1]), (pred_boxes[idx][2], pred_boxes[idx][3])])
                filtered_classes.append(class_name)
                filtered_scores.append(score)
    
    return filtered_boxes, filtered_classes, filtered_scores


def object_detection_api(img_path, threshold=0.5, rect_th=2, text_size=1, text_th=1):
    """
    object_detection_api
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
    """
    boxes, pred_cls, pred_score = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        pt1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
        pt2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
        
        # Add detection confidence percentage to the label
        label = f"{pred_cls[i]}: {int(pred_score[i] * 100)}%"
        cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Gọi hàm object_detection_api từ bên ngoài
object_detection_api('./anh.jpg', threshold=0.8)
