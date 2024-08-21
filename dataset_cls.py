import os
from PIL import Image
from torch.utils.data import Dataset

# Custom Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("L")  # 흑백 이미지로 불러옴
        mask = Image.open(mask_path).convert("L")  # 흑백 이미지로 불러옴

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

