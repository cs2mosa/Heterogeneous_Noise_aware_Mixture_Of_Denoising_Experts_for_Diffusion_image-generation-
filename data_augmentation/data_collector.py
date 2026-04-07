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

def get_dataloader_path(model_config: dict, data_dir: str = "/kaggle/input/YOUR_DATASET_FOLDER"):
    """
    High-Performance Dataloader for Large-Scale Diffusion Training.
    Outputs: float32 tensors in the range[-1.0, 1.0]
    """
    # 1. Image Preprocessing Pipeline
    transform = transforms.Compose([
        # Resize the shortest edge to 256, then take a perfect center square
        transforms.Resize(model_config["data_img_res"], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(model_config["data_img_res"]),

        # Convert to Tensor (scales[0, 255] integer pixels to [0.0, 1.0] floats)
        transforms.ToTensor(),

        # Shift to [-1.0, 1.0] (Crucial for VAE inputs)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Dataset Setup
    # Assumes your data is inside a folder structure like: data_dir/class_name/image.jpg
    # If there are no subfolders, put the images in one subfolder inside data_dir!
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    print(f"Dataset Loaded: {len(dataset)} images found.")

    # 3. High-Throughput DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,  # Randomize the trajectory
        num_workers=4,  # Kaggle generally handles 2 to 4 workers optimally
        pin_memory=True,  # Speeds up CPU-to-GPU transfer
        drop_last=True,  # Ensures every batch is exactly batch_size (prevents shape errors)
        prefetch_factor=2  # Pre-loads the next 2 batches in the background
    )

    return dataloader