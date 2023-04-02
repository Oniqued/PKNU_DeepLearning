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

# 유사한 구조의 네트워크를 다르게 정의할 수도 있다.
class NeuralNetwork2(nn.Module):  # 512와 1024 차이 뿐 다른건 같은 코드
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.fc1 = nn.Linear(img_height * img_width * num_channels, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input image / x.size -> batch size / -1 -> undefined // 그냥 이런것도 있다
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# NeuralNetwork의 인스턴스를 생성하고 이를 device로 이동한 뒤, 구조를 출력한다.
model = NeuralNetwork().to(device) # 필요한 작업... CPU 쓸건지 GPU 쓸건지
print(model)

# 랜덤 픽셀로 구성된 2개의 흑백이미지를 nn.Softmax를 이용해서 출력해본다.
X = torch.rand(2, 1, 28, 28, device=device)  # 2 random grayscale images / 픽셀값은 랜덤인 2개의 이미지
logits = model(X)
print('logits: {}'.format(logits))
pred_probab = nn.Softmax(dim=1)(logits) # softMax가 적용되는 함수가 1차원 이기 때문에 dim=1
print('pred_probab: {}'.format(pred_probab))

y_pred = pred_probab.argmax(1) # argmax? 10개의 텐서 중 몇번째 값이 최댓값인지 찾아주는 함수 '(1)'은 '축'이라는 뜻(?) //아래 Predicted class: tensor([0, 8]) 는 최대값이 각각 이미지 별로 0과 8이다.
print(f"Predicted class: {y_pred}")

# nn.Softmax ?
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
pred_probab

# 모델 계층
# 3개의 랜덤 이미지의 계층을 출력해보자
input_images = torch.rand(3, 1, 28, 28) # 2 random grayscale images
print(input_images)
print(input_images.size())

# nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_images)
print(flat_image.size())

# nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20) # 노드가 20개인 리니어 레이어 (곧, 출력 벡터의 사이즈즈)
hidden1 = layer1(flat_image)
print(hidden1.size())
print(hidden1)

# nn.ReLU
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}") # 음수는 0으로, 양수는 그대로 출력된 것을 볼 수 있다.

# nn.Sequential
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_images = torch.rand(3, 1, 28, 28)
logits = seq_modules(input_images)
print(logits)

