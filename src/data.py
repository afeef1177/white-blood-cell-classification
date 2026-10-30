import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def build_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

def create_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=2):
    train_tf, eval_tf = build_transforms(image_size)
    paths = {s: os.path.join(data_dir, s) for s in ("train", "val", "test")}
    for split, path in paths.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing {path}; expected {split}/<class_name>/images")
    train_ds = datasets.ImageFolder(paths["train"], transform=train_tf)
    val_ds = datasets.ImageFolder(paths["val"], transform=eval_tf)
    test_ds = datasets.ImageFolder(paths["test"], transform=eval_tf)
    class_map = train_ds.class_to_idx
    for name, ds in (("val", val_ds), ("test", test_ds)):
        if ds.class_to_idx != class_map:
            raise ValueError(f"Class folders in {name} do not match train: {ds.class_to_idx}")
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, train_ds.classes
