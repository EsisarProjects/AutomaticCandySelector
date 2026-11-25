"""ConvNeXt-Tiny construction and checkpoint compatibility validation."""
from pathlib import Path
import torch
from torch import nn
from torchvision import models

ARCHITECTURE = 'convnext_tiny'


def create_convnext(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    original = model.classifier[2]
    model.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(original.in_features, num_classes))
    return model


def read_checkpoint(checkpoint_path: Path) -> dict:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'ConvNeXt checkpoint not found: {path}')
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(bundle, dict) or 'classes' not in bundle or 'model_state_dict' not in bundle:
        raise ValueError('ConvNeXt checkpoint requires classes and model_state_dict')
    classes = bundle['classes']
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
        raise ValueError('Invalid ConvNeXt checkpoint class list')
    if bundle.get('architecture', ARCHITECTURE) != ARCHITECTURE:
        raise ValueError('Checkpoint does not contain ConvNeXt-Tiny weights')
    mapping = bundle.get('class_to_idx', {name: i for i, name in enumerate(classes)})
    if mapping != {name: i for i, name in enumerate(classes)}:
        raise ValueError('ConvNeXt checkpoint class_to_idx mismatches class ordering')
    return bundle


def load_convnext_model(checkpoint_path, device, class_names=None):
    bundle = read_checkpoint(Path(checkpoint_path))
    classes = bundle['classes']
    if class_names is not None and list(class_names) != classes:
        raise ValueError('Configured classes do not match the ConvNeXt checkpoint')
    model = create_convnext(len(classes), pretrained=False)
    model.load_state_dict(bundle['model_state_dict'], strict=True)
    model.to(device).eval()
    return model
