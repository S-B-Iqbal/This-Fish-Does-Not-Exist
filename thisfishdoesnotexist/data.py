from torch.utils.data import Dataset
from PIL import Image
class FishDataset(Dataset):
    """Class for loading an Image."""
    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.images.iloc[idx])

        if self.transform:
            img = self.transform(img)
            label = self.labels.iloc[idx]
        return img, label
