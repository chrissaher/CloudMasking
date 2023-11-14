import os
import cv2
from PIL import Image
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class CloudDataset(Dataset):
    """
    ToDo: Add augmentation, measure class imbalance, visualize some samples.
    """
    CLASSES = ["background", "cloud"]

    def __init__(self, tiles_dir: Path, random_resize_crop: tuple=None, max_dataset_len: int=None):
        """
        tiles_dir: Path to where tiles are stored
        random_resize_crop: Applies random resize cropping to input image. If None, this augmentation is skipped.
        max_dataset_len: Maximum number of tiles to load for a dataset. Useful for fast iterations
        """
        images_dir = tiles_dir / "images"
        masks_dir = tiles_dir / "masks"
        dataset_total_len = len(os.listdir(str(images_dir)))
        dataset_actual_len = max_dataset_len if max_dataset_len is not None else dataset_total_len
        if dataset_actual_len > dataset_total_len:
            print(f"'max_dataset_len' is greater than the dataset length. Loading {dataset_total_len} files instead.")
            dataset_actual_len = dataset_total_len

        self.images = list()
        self.masks = list()

        image_transforms = list()
        mask_transforms = list()

        image_transforms.extend([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        if random_resize_crop is not None:
            image_transforms.append(v2.RandomResizedCrop(size=random_resize_crop, antialias=True))
            mask_transforms.append(v2.RandomResizedCrop(size=random_resize_crop, antialias=True))

        image_transforms.extend([
            v2.ToPureTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        mask_transforms.extend([
            v2.ToPureTensor(),
        ])

        image_transforms = v2.Compose(image_transforms)
        mask_transforms = v2.Compose(mask_transforms)

        for image_path in images_dir.iterdir():
            if len(self.images) == dataset_actual_len:
                break
            mask_path = masks_dir / image_path.name

            image = None
            mask = None

            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = tv_tensors.Mask(mask / 255, dtype=torch.int64)
            mask = mask_transforms(mask)
            mask = self._one_hot_encode(mask)

            self.images.append(image_transforms(image))
            self.masks.append(mask.float())

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return len(self.images)

    def _one_hot_encode(self, mask):
        H, W = mask.shape
        one_hot = torch.zeros((len(self.CLASSES), H, W), dtype=mask.dtype, device=mask.device)
        one_hot.scatter_(0, mask.unsqueeze(0), 1)
        return one_hot
