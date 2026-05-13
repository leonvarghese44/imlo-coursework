import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from model import SimpleCNN


class PetTrimapDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, image_size):
        self.base_dataset = base_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, target = self.base_dataset[index]
        label, trimap = target

        image = image.convert("RGB")
        trimap = trimap.convert("L")

        image = F.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )

        trimap = F.resize(
            trimap,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        image_tensor = F.to_tensor(image)

        image_tensor = F.normalize(
            image_tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        trimap_array = np.array(trimap)

        trimap_channel = np.zeros_like(trimap_array, dtype=np.float32)
        trimap_channel[trimap_array == 3] = 0.5
        trimap_channel[trimap_array == 1] = 1.0

        trimap_tensor = torch.from_numpy(trimap_channel).unsqueeze(0)

        combined = torch.cat([image_tensor, trimap_tensor], dim=0)

        return combined, label


def calculate_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_size = 192
    batch_size = 32

    base_test_dataset = OxfordIIITPet(
        root="data",
        split="test",
        target_types=["category", "segmentation"],
        download=True,
        transform=None,
        target_transform=None
    )

    test_dataset = PetTrimapDataset(
        base_dataset=base_test_dataset,
        image_size=image_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = SimpleCNN(num_classes=37).to(device)

    model.load_state_dict(
        torch.load("model.pth", map_location=device)
    )

    test_accuracy = calculate_accuracy(model, test_loader, device)

    print("Test images:", len(test_dataset))
    print("Test accuracy:", test_accuracy)
    print("Test accuracy percentage:", test_accuracy * 100)


if __name__ == "__main__":
    test()