import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from tqdm import tqdm

from model import SimpleCNN


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

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_dataset = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transform
    )

    train_eval_dataset = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=eval_transform
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