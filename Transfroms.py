import torch
from torchvision import datasets
import torchvision.transforms as transforms

train_ds = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((36, 36)),
        transforms.RandomCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
    target_transform=transforms.Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_ds = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
    target_transform=transforms.Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

X, y = train_ds[0]
print(X)
print(X.shape)
print(X.mean(), X.std(), X.max(), X.min())

print(y)
print(y.shape)

# TorchVision이 제공하는 미리 정의된 transform 기능들
transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Resize((36,36)),
                       transforms.RandomCrop((32, 32)),
                       transforms.RandomHorizontalFlip(),
                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),

# Lambda 변형
target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

