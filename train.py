import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from model import SimpleCNN


class PetTrimapDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, image_size, train):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.train = train

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

        if self.train:
            if random.random() < 0.5:
                image = F.hflip(image)
                trimap = F.hflip(trimap)

            angle = random.uniform(-8, 8)

            image = F.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )

            trimap = F.rotate(
                trimap,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=2
            )

            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            saturation = random.uniform(0.9, 1.1)

            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_size = 192
    batch_size = 32
    epochs = 30

    lr_schedule = (
        [0.001] * 24 +
        [0.0005] * 4 +
        [0.0002] * 2
    )

    train_base_dataset = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types=["category", "segmentation"],
        download=True,
        transform=None,
        target_transform=None
    )

    train_dataset = PetTrimapDataset(
        base_dataset=train_base_dataset,
        image_size=image_size,
        train=True
    )

    train_eval_dataset = PetTrimapDataset(
        base_dataset=train_base_dataset,
        image_size=image_size,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = SimpleCNN(num_classes=37).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimiser = optim.Adam(
        model.parameters(),
        lr=lr_schedule[0],
        weight_decay=0.0001
    )

    print("Training images:", len(train_dataset))
    print("Validation images: 0")
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Image size:", image_size)
    print("Input channels: 4")
    print("Trimap encoding: background 0.0, border 0.5, pet 1.0")
    print("Learning rate schedule:", lr_schedule)

    for epoch in range(epochs):
        current_lr = lr_schedule[epoch]

        for param_group in optimiser.param_groups:
            param_group["lr"] = current_lr

        model.train()

        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        train_accuracy = calculate_accuracy(model, train_eval_loader, device)

        print("Epoch:", epoch + 1)
        print("Learning rate:", current_lr)
        print("Loss:", average_loss)
        print("Train accuracy:", train_accuracy)

    torch.save(model.state_dict(), "model.pth")

    print("Saved model.pth")
    print("Finished training")


if __name__ == "__main__":
    train()