import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import json
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoDataset(Dataset):
    def __init__(self, annotations_file, images_folder, transforms=None, image_ids=None):
        with open(annotations_file, "r") as f:
            self.coco_data = json.load(f)
        self.images_folder = images_folder
        self.transforms = transforms
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Group annotations by image_id for easier access
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Filter images by provided image_ids
        if image_ids is not None:
            self.image_info = {k: v for k, v in self.image_info.items() if k in image_ids}

        self.total_images = len(self.image_info)  # Total number of images

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        # Lấy thông tin ảnh theo chỉ mục idx từ danh sách ảnh (mảng)
        img_info = list(self.image_info.values())[idx]  # Truy cập trực tiếp vào giá trị theo index
        img_path = os.path.join(self.images_folder, img_info['file_name'])
        
        # Hiển thị tiến trình
        print(f"Reading image {idx + 1}/{self.total_images}: {img_info['file_name']}")
        
        image = Image.open(img_path).convert("RGB")

        anns = self.img_to_anns.get(img_info['id'], [])
        boxes = []
        labels = []

        for ann in anns:
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            # Nếu không có annotations, bỏ qua ảnh này và lấy ảnh tiếp theo
            print(f"No annotations found for image {img_info['file_name']}. Skipping.")
            return self.__getitem__((idx + 1) % self.total_images)  # Lấy ảnh kế tiếp

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


# Function to get the model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    model.train()
    loss_values = []
    mAP_values = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
                
            epoch_loss += losses.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        mAP = evaluate(model, test_loader, device)
        mAP_values.append(mAP)
        print(f"Epoch {epoch+1}/{num_epochs}, mAP: {mAP:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, num_epochs+1), loss_values, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(range(1, num_epochs+1), mAP_values, label='Validation mAP', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def evaluate(model, data_loader, device):
    model.eval()
    coco_gt = COCO(annotations_file)
    coco_dt_list = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            # Chuyển đổi outputs sang định dạng COCO
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    coco_dt_list.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [xmin, ymin, width, height],
                        "score": score
                    })

    # Tạo COCO format cho dự đoán
    coco_dt = coco_gt.loadRes(coco_dt_list)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]
    return mAP

# Main script
if __name__ == "__main__":
    annotations_file = "./annotations.json"
    images_folder = "./dataset-animal"
    num_classes = len(json.load(open(annotations_file))['categories']) + 1 

    with open(annotations_file, "r") as f:
        coco_data = json.load(f)
    image_ids = [img['id'] for img in coco_data['images']]
    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

    # Dataset and DataLoader
    train_dataset = CocoDataset(annotations_file, images_folder, transforms=torchvision.transforms.ToTensor(), image_ids=train_ids)
    test_dataset = CocoDataset(annotations_file, images_folder, transforms=torchvision.transforms.ToTensor(), image_ids=test_ids)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4,  
        collate_fn=lambda x: tuple(zip(*x))
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4,  
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Prepare model and optimizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Train model
    train_model(model, train_loader, test_loader, optimizer, device, num_epochs=10)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fasterrcnn_finetuned.pth")
