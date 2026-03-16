from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset

def get_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg["data_img_res"], cfg["data_img_res"])),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Map to [-1, 1]
    ])
    print("Downloading/Loading Flowers102 Dataset splits...")
    train_set = datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
    val_set = datasets.Flowers102(root='./data', split='val', download=True, transform=transform)
    test_set = datasets.Flowers102(root='./data', split='test', download=True,
                                   transform=transform)

    full_dataset = ConcatDataset([train_set, val_set, test_set])
    print(f"Total images loaded: {len(full_dataset)}")

    return DataLoader(
        full_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )