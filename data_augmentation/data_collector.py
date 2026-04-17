from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import torch
from PIL import Image
from datasets import load_dataset

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

NATURE_KEYWORDS = [
    "lake", "forest", "mountain", "river", "landscape", "tree",
    "nature", "meadow", "waterfall", "valley", "wilderness",
    "shore", "woodland", "scenery", "outdoor", "canyon", "glacier"
]

def get_dataloader_hf(model_config: dict):
    # 1. Transform for 512x512 training
    transform = transforms.Compose([
        transforms.Resize(model_config["data_img_res"], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(model_config["data_img_res"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Load via WebDataset tar shards (512x512 subset, 46 shards)
    print("Loading text-to-image-2M via WebDataset shards...")
    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
    urls = [base_url.format(i=i) for i in range(46)]

    dataset = load_dataset(
        "webdataset",
        data_files={"train": urls},
        split="train",
        streaming=True
    )

    # 3. Filter for nature captions only
    def is_nature_sample(example):
        prompt = example.get("json", {})
        if isinstance(prompt, str):
            import json
            try:
                prompt = json.loads(prompt)
            except Exception:
                return False
        caption = prompt.get("prompt", "").lower()
        return any(kw in caption for kw in NATURE_KEYWORDS)

    dataset = dataset.filter(is_nature_sample)
    print("Nature keyword filter applied.")

    # 4. Preprocessing — image is in 'jpg', caption is in 'json'['prompt']
    def preprocess_fn(examples):
        images = []
        for img in examples["jpg"]:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            images.append(transform(img.convert("RGB")))

        captions = []
        for j in examples["json"]:
            if isinstance(j, str):
                import json as jsonlib
                try:
                    j = jsonlib.loads(j)
                except Exception:
                    j = {}
            captions.append(j.get("prompt", ""))

        examples["pixel_values"] = images
        examples["text"] = captions
        return examples

    transformed_dataset = dataset.map(preprocess_fn, batched=True)

    # 5. Collate function
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        prompts = [example["text"] for example in examples]
        return {"pixel_values": pixel_values, "prompts": prompts}

    # 6. Buffer shuffle (required for streaming/IterableDataset)
    shuffled_dataset = transformed_dataset.shuffle(seed=42, buffer_size=2000)

    return DataLoader(
        shuffled_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,       # Must be False for streaming IterableDataset
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )