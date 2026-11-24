import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
IMAGE_SIZE = 224  # ConvNeXt chuẩn với ảnh 224x224
EPOCHS = 10
LEARNING_RATE = 0.0002

DATA_DIR = ROOT / "datasets" / "classification" / "raw"

class ACSFolderDataset(Dataset):
    def __init__(self, root_dir, session="train", transform=None):
        self.samples = []
        self.transform = transform
        self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        target_splits = [session]
        if session == "train": target_splits = ["train", "S01"]
        elif session == "test": target_splits = ["test", "val", "S02", "S03"]

        for candy_type in self.classes:
            type_dir = root_dir / candy_type
            if not type_dir.is_dir(): continue
            for color_dir in type_dir.iterdir():
                if not color_dir.is_dir(): continue
                for split_name in target_splits:
                    session_dir = color_dir / split_name
                    if session_dir.is_dir():
                        for img_path in session_dir.glob("*.*"):
                            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                                self.samples.append((img_path, self.class_to_idx[candy_type]))
                            
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ACSFolderDataset(root_dir=DATA_DIR, session="train", transform=train_transforms)
test_dataset = ACSFolderDataset(root_dir=DATA_DIR, session="test", transform=test_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)

# SỬ DỤNG CONVNEXT-TINY (Xịn, hiện đại và ổn định)
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
model.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[2].in_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

model_path = ROOT / "models" / "candy_convnext_tiny.pth"
model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_dataset.classes,
    "class_to_idx": train_dataset.class_to_idx
}, model_path)
print(f"-> [Đã lưu model ConvNeXt-Tiny vào: {model_path}]")
