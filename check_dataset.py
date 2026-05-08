import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="category",
    download=True,
    transform=transform
)

test_dataset = OxfordIIITPet(
    root="data",
    split="test",
    target_types="category",
    download=True,
    transform=transform
)

print("Trainval size:", len(train_dataset))
print("Test size:", len(test_dataset))

image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Label:", label)

print("CUDA available:", torch.cuda.is_available())