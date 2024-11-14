import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as F
from torchvision import transforms
import os
import cv2
import json
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        
        with open(label_file, 'r') as f:
            anno_dict = json.load(f)
        
        self.image_map = anno_dict['images']
        self.annotations = anno_dict['annotations']

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        
        image_id = str(anno['image_id'])
        if image_id not in self.image_map:
            return None

        img_name = self.image_map[image_id]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        keypoints = []
        masks = []
        
        for anno in self.annotations:
            if anno['image_id'] == int(image_id):
                bbox = anno['bbox']
                keypoint = anno.get('keypoints', [])
                
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                boxes.append([x_min, y_min, x_max, y_max])
                
                labels.append(anno['category_id'])
                keypoints.append(keypoint)
                masks.append(torch.zeros((256, 256), dtype=torch.uint8))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).view(-1, 3)
        masks = torch.stack(masks)

        target = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints,
            "masks": masks
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.annotations)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch)) if len(batch) > 0 else ([], [])

transform = transforms.Compose([transforms.ToTensor()])
dataset = CustomDataset(
    image_dir="animalpose_keypoint_new/animalpose_image_part2",
    label_file="animalpose_keypoint_new/keypoints.json",
    transforms=transform
)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
num_classes = len(set(anno['category_id'] for anno in dataset.annotations)) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes
)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    train_loss = 0.0
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)", unit="batch")
    
    for images, targets in train_progress_bar:
        if len(images) == 0:
            continue

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        train_progress_bar.set_postfix(loss=losses.item())

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", unit="batch"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            val_loss += losses.item()

    avg_val_loss = val_loss / len(val_loader)

    test_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Test", unit="batch"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            test_loss += losses.item()

    avg_test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    lr_scheduler.step()

torch.save(model.state_dict(), "mask_rcnn_model.pth")
