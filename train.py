import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from tqdm import tqdm

from model import SimpleCNN


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
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    full_trainval_dataset = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=transform
    )

    train_size = int(0.8 * len(full_trainval_dataset))
    val_size = len(full_trainval_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_trainval_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    model = SimpleCNN(num_classes=37).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1
    best_val_accuracy = 0.0

    for epoch in range(epochs):
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

        train_accuracy = calculate_accuracy(model, train_loader, device)
        val_accuracy = calculate_accuracy(model, val_loader, device)
        average_loss = running_loss / len(train_loader)

        print("Epoch:", epoch + 1)
        print("Loss:", average_loss)
        print("Train accuracy:", train_accuracy)
        print("Validation accuracy:", val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "model.pth")
            print("Saved model.pth")

    print("Finished training")
    print("Best validation accuracy:", best_val_accuracy)


if __name__ == "__main__":
    train()