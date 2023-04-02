import torch
from torchvision import datasets
import torchvision.transforms as transforms

train_ds = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((36, 36), antialias=True),
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
        transforms.Resize((32, 32), antialias=True),
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
                       transforms.Resize((36,36), antialias=True),
                       transforms.RandomCrop((32, 32)),
                       transforms.RandomHorizontalFlip(),
                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),

# Lambda 변형
target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

# # 사례: 얼굴 포즈(facial pose) 데이터셋
import os
import torch
import pandas as pd
from skimage import io, transform
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

data_dir = 'content/faces'
# 얼굴 랜드마크 정보를 저장하는 주석 파일 열기
landmarks_frame = pd.read_csv(data_dir + '/face_landmarks.csv')
landmarks_frame

# 66번 행 상세보기
n = 66
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

# 랜드마크 점 표시해서 디스플레이
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.show()

show_landmarks(Image.open(os.path.join(data_dir, img_name)), landmarks)

# 이 데이터셋을 표현하는 커스텀 데이터셋 클래스 FaceLandmarksDataset 정의하기
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = np.array(Image.open(img_name)) # PIL Image를 numpy array로 변환한다.

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# 데이터를 추출하여 디스플레이

face_dataset = FaceLandmarksDataset(csv_file=data_dir + '/face_landmarks.csv',
                                    root_dir=data_dir)

# fig = plt.figure()

for i, sample in enumerate(face_dataset):
    # sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)
    show_landmarks(sample['image'], sample['landmarks'])

    # ax = plt.subplot(1, 4, i + 1)
    # plt.tight_layout()
    # ax.set_title('Sample #{}'.format(i))
    # ax.axis('off')
    # show_landmarks(**sample)

    if i == 3:
        # plt.show()
        break