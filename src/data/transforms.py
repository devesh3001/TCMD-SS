from torchvision import transforms


def image_transform(size: tuple[int, int] = (224, 224)) -> transforms.Compose:
    # Deterministic preprocessing avoids introducing augmentation during inspection.
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
