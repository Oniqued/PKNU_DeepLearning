import torch
import numpy as np

# 데이터로부터 직접 새성하기
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

# NumPy 배열로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array) # from_numpy 메서드로 Tensor를 생성
print(x_np)
print(x_np.dtype)

x_np2 = torch.tensor(np_array, dtype=float) # torch.tensor 생성자로 Tensor를 생성할 수도 있음. dtype을 지정 가능
print(x_np2)
print(x_np2.dtype)

np_array[0,0] = 0 # 배열의 값을 수정한다
print(x_np)       # from_numpy로 생성한 경우 배열의 데이터를 공유한다
print(x_np2)      # tensor 생성자로 생성한 경우 배열의 데이터를 공유하지 않는다

# 모양(shape)을 지정하여 텐서 생성하기
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor}\n")
print(f"Ones Tensor: \n {ones_tensor}\n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 다른 텐서로부터 생성하기
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(x_ones)
x_zeros = torch.zeros_like(x_data) # x_data의 속성을 유지합니다.
print(x_zeros)
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(x_rand)

# 텐서의 기본 속성(attributes)
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 텐서 연산 (Operation)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
# NumPy 방식의 인덱싱과 슬라이싱
tensor = torch.tensor([[1,2,3,4],
                       [5,6,7,8],
                       [9,10,11,12],
                       [13,14,15,16]
                       ])
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")    # ...은 앞쪽의 모든 차원에 대해서 "모든 범위"임을 의미한다. 3차원 이상의 다차원 배열을 다룰 때 유용한 표현이다.
tensor[:,1] = 0
print(tensor)

# 텐서 합치기
tensor = torch.tensor([[1,2,3,4],
                       [5,6,7,8],
                       [9,10,11,12]])
t0 = torch.cat([tensor, tensor])
print(t0)
t1 = torch.cat([tensor, tensor], dim = 1) # dim을 지정하여 어떤 축으로 연결할지 지정한다
print(t1)

# torch.stack
t2 = torch.stack([tensor, tensor])
print(t2.shape)
print(t2)

#axis 인자
t3 = torch.stack([tensor, tensor], axis=1)
print(t3.shape)
print(t3)

# 텐서 모양 변경하기
a = torch.arange(12)
a
a1 = torch.reshape(a, (3,4))
a1

b = torch.reshape(a, (-1,2,3)) # 크기가 유추 가능할 경우 -1로 표시 가능
print(b)
print(b.shape)

#squeeze, unsqueeze 메서드
x = torch.rand(1,1,20,128)
x = x.squeeze() # [1,1,20,128] -> [20,128]
print(x.shape)

x2 = torch.rand(1,1,20,128)
x2 = x2.squeeze(dim=1) # [1,1,20,128] -> [1,20,128]
print(x2.shape)

x = torch.rand([2,3])
x1 = torch.unsqueeze(x,0)
x1.shape
