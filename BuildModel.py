import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# cuda를 사용할 수 있는지 확인하고 그렇지 않으면 cpu를 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 클래스 정의하기
img_height = 28     # 이미지의 높이
img_width = 28      # 이미지의 너비
num_channels = 1    # 흑백 이미지이므로 1
num_classes = 10    # 분류할 이미지의 클래스는 10가지

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # 2차원 모델을 1차원 모델로 변환
        self.linear_relu_stack = nn.Sequential( # 3개의 리니어 모듈로 구성
            nn.Linear(img_height*img_width*num_channels, 512), # 노드 갯수는 512개 (출력 사이즈도 512가 됨)
            nn.ReLU(),
            nn.Linear(512, 512), # 노드 갯수도 512개
            nn.ReLU(),
            nn.Linear(512, num_classes), # 마지막 레이어는 Activation fun 없음
        )

    def forward(self, x): # 입력x : 이미지
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

