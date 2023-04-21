# 하단의 코드는 yunjey의 pytorch-tutorial을 바탕으로 주석 해석한 것임을 밝힘 #
# 출처: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py #
# 인공지능과 보안기술 강의 과제용으로 튜토리얼 코드를 가져와 영문 주석 번역 및 모르는 부분 추가하였음 #

import torch  # torch import
import torchvision # cnn(convolutional neural network)을 위한 torchvision import
import torch.nn as nn # nn(신경망)기능 import
import numpy as np # numpy 모듈 np라는 이름으로 import
import torchvision.transforms as transforms # 이미지 변환을 위해 torchvision.transforms 기능을 transforms 라는 이름으로 import


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189)


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# tensor 만들기
x = torch.tensor(1., requires_grad=True)
# 기울기를 구하기 위해 require_grad 값을 True로 줌
# torch.tensor는 값 복사를 사용하여 새로운 텐서 자료형 인스턴스 생성
# 즉, 1. 값 복사하여 기울기 여부 True로 x에 텐서를 하나 생성하는 것임
w = torch.tensor(2., requires_grad=True)
# w와 b역시 x와 마찬가지로 각각의 값을 복사하여 기울기 True로 텐서를 생성
b = torch.tensor(3., requires_grad=True)

# 계산 그래프 작성
y = w * x + b    # y = 2 * x + 3 -> 위에서 만든 텐서의 값

# 경사도 계산
y.backward() # backward = gradient(변화도) 자동계산
# autograd를 사용하여 역전파 단계 계산

# 경사도 출력
print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# (10, 3)과 (10, 2)의 모양에 랜덤한 수를 갖는 tensor 두 개를 생성
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# fully connected layer 만들기
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build loss function and optimizer.
# Loss function(손실함수): 모델을 통해 생성된 결과 값과 실제로 발생하기를 원했던 값간의 차이를 계산하는 함수.
criterion = nn.MSELoss() # 간단한 손실함수. 입력과 정답 사이의 평균 제곱 오차를 계산
# 최적화를 위해 SGD(Stochastic Gradient Descent, 확률적 경사하강법)사용
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
# -> 손실 함수의 값을 계산하기 위한 것
pred = linear(x)

# Compute loss.
# 손실 계산
loss = criterion(pred, y)
print('loss: ', loss.item()) # 손실값 출력

# Backward pass.
# -> learnable parameters의 gradients를 계산
loss.backward()

# 경사도 출력
# 하단의 weight.grad와 bias.grad에 대한 코드는 아무리 찾아도 위의 경사도 출력만 나오고
# 해설이 없어 가중치 기울기 및 편향 기울기를 표현하는 것으로 이해하였다.
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
# 1단계 경사 강하
optimizer.step()

# 낮은 레벨에서의 경사 하강도 또한 수행할 수 있음
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# 1단계 경사 강하 후 손실 출력
pred = linear(x)
loss = criterion(pred, y) # 손실 계산 파트
print('loss after 1 step optimization: ', loss.item()) # 경사 강하 후 값을 출력


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# numpy 배열 생성
# numpy는 tensor와 동기화 된다고 적혀있었음(참고)
x = np.array([[1, 2], [3, 4]])

# 생성된 numpy 배열을 torch tensor로 변환
y = torch.from_numpy(x)

# torch tensor를 numpy 배열로 변환
z = y.numpy()


# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

# CIFAR-10 데이터셋을 다운로드하고 구성
# CIFAR-10이란 기계학습에서 흔히 사용되는 벤치마크 문제.
# RGB 32x32 픽셀 이미지를 10개 카테고리로 분류하는 것이 목표
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# 데이터 쌍을 하나씩 가져온다(디스크에서 데이터를 읽어옴_CIFAR-10)
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (매우 간단한 방법으로 queue와 thread를 제공).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# 반복이 시작되면, 큐와 쓰레드가 파일에서 데이터를 불러오기 시작
data_iter = iter(train_loader)

# Mini-batch 이미지와 라벨
images, labels = data_iter.next()

# data loader의 실제 사용은 아래와 같음
for images, labels in train_loader:
    # 훈련 코드를 여기에 작성해야 함
    pass


# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #

# 각 개인의 커스텀 데이터셋을 아래에 작성
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. 파일 경로 또는 파일 이름의 리스트를 초기화
        pass
    def __getitem__(self, index):
        # TODO
        # 1. 파일에서 데이터 하나를 읽어옴(예. numpy.fromfile, PIL.Image.open을 사용)
        # 2. 데이터 사전처리(예. torchvision.Transform)
        # 3. 데이터 쌍 반환(예. 이미지와 라벨)
        pass
    def __len__(self):
        # 0을 당산의 데이터셋의 총 사이즈로 바꿔야함
        return 0

# 그런 다음 미리 구축된 data loader를 사용할 수 있음
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# 미리 훈련된 ResNet-18을 다운로드하고 불러옴
# ResNet-18은 ILSVRC에서 2015년 우승한 심층 신경망 모델.
resnet = torchvision.models.resnet18(pretrained=True)

# 모델의 상단 레이어만 미세 조정하길 원한다면, 아래와 같이 설정
# Fine-tuning이란, 모델의 파라미터를 미세하게 조정하는 행위.
# 특히 딥러닝에서는 이미 존재하는 모델에 추가 데이터를 투입하여 파라미터를 업데이트 하는 것을 말함
for param in resnet.parameters():
    param.requires_grad = False

# 미세 조정을 위해 상단 레이어를 교체
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 = 예시

# Forward pass(손실함수의 값 계산).
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# 전체 모델을 저장하고 불러오기
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# 모델 파라미터만 저장하고 불러오기(권장됨).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))