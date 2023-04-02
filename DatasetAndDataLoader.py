import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 데이터셋 불러오기
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 데이터셋을 순회하고 시각화하기
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8))
cols, rows = 3,3
for i in range(1,cols*rows+1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray") # squeeze()는 shape이 (1,28,28) 텐서를 (28,28)인 텐서로 변환
plt.show()

# 사용자 정의 데이터셋 만들기
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset): #세가지 메서드 구현 필요
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): # 필요한 것들 준비
            self.img_labels = pd.read_csv(annotations_file) # csv파일을 주면 읽어서 내부적으로 테이블 형태의 DataStructure를 만들어줌
            self.img_dir = img_dir  # 이미지 저장소
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):  # 데이터 셋 길이 리턴
            return len(self.img_labels)

        def __getitem__(self, idx):  # 인자로 하나의 인덱스를 받아서 하나의 데이터 중에 인덱스 번호의 데이터 중에 나머지 하나를 리턴
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # <<csv파일임
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

# DataLoader로 학습용 데이터 준비하기
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# DataLoader를 통해 순회하기
# 이미지와 정답(label)을 표시합니다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")