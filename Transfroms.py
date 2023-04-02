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

# 이미지 크기를 균일하게 만들기, 랜드마크 포인트 좌표 변경
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        img = Image.fromarray(image)
        img = img.resize((new_h, new_w), resample=0)
        img = np.array(img)
        # img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple): Desired output size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

# Rescale, RandomCrop 테스트
scale = Rescale((256, 256))
crop = RandomCrop((128, 128))
composed = transforms.Compose([Rescale((256, 256)),
                               RandomCrop((224, 224))])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[66]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()

# numpy 배열 형태의 이미지를 Tensor로 변환하고 정규화 하는 코드
class MyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.img_tensorfier = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize가 하는 일은 무엇인가

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = self.img_tensorfier(image)

        # image = image.transpose((2, 0, 1))
        return {'image': image,  # torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

transformed_dataset = FaceLandmarksDataset(csv_file=data_dir + '/face_landmarks.csv',
                                           root_dir=data_dir,
                                           transform=transforms.Compose([
                                               Rescale((256, 256)),
                                               RandomCrop((224, 224)),
                                               MyToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())
    print(sample['image'].min(), sample['image'].max())

    if i == 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    grid = grid / 2 + 0.5
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

# if you are using Windows, uncomment the next line and indent the for loop.
# you might need to go back and change "num_workers" to 0.

# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        # show_landmarks(sample_batched['image'][0].numpy().transpose((1, 2, 0)), sample_batched['landmarks'][0])
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break