"""Identical geometric preprocessing for ConvNeXt training and inference."""
from PIL import Image, ImageOps
from torchvision import transforms


class RGBWithExif:
    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image).convert('RGB')


class Letterbox:
    def __init__(self, size: int, color=(255, 255, 255)):
        self.size = int(size)
        self.color = tuple(color)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), self.color)
        canvas.paste(image, ((self.size - image.width)//2, (self.size - image.height)//2))
        return canvas


def build_transform(settings: dict, training: bool = False):
    mode = settings.get('resize_mode', 'stretch')
    if mode == 'stretch':
        steps = [RGBWithExif(),
                 transforms.Resize((settings['image_size'], settings['image_size']))]
    elif mode == 'letterbox':
        steps = [Letterbox(settings['image_size'], settings['pad_color'])]
    else:
        raise ValueError(f'Unsupported resize_mode: {mode}')
    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180, fill=tuple(settings['pad_color'])),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=settings['normalize_mean'], std=settings['normalize_std']),
    ])
    return transforms.Compose(steps)
