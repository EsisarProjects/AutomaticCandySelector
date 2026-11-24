from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

def load_convnext_model(checkpoint_path, device, class_names):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    num_classes = len(class_names)
    
    # Khởi tạo khung ConvNeXt-Tiny
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[2].in_features, num_classes)
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
