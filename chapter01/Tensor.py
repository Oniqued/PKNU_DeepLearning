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

# axis 인자
t3 = torch.stack([tensor, tensor], axis=1)
print(t3.shape)
print(t3)

# 산술연산
# 텐서와 스칼라간의 기본적인 산술 연산을 지원한다
ones = torch.zeros(2,2) + 1
twos = torch.ones(2,2)*2
threes = (torch.ones(2,2)*7-1)/2
fours = twos**2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

fives = ones + fours
print(fives)

# 요소별 곱(element-wise product)을 계산한다. dozens1, dozens2는 같은 값을 갖는다.
dozens1 = threes * fours
dozens2 = threes.mul(fours)
print(dozens1)
print(dozens2)

#두 텐서 간의 행렬곱(matrix multiplication)을 계산한다. y1, y2는 같은 값을 갖는다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(y1)
print(y2)

#Broadcasting
A = torch.rand(2, 4)
B = torch.ones(1, 4) * 2
print(A)
print(B)
C = A + B
print(C)
D = A * B
print(D)

# 기타 산술연산
# common functions
a = torch.rand(2, 4) * 2 - 1
print(a)
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))  # 올림 연산
print(torch.floor(a)) # 내림 연산
print(torch.clamp(a, -0.5, 0.5)) # -0.5 이하는 -0.5로, 0.5 이상은 0.5로 변환

# comparisons:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

# reductions:
print('\nReduction ops:')
print(torch.max(d))        # returns a single-element tensor
print(torch.min(d))        # returns a single-element tensor
print(torch.mean(d))       # average
print(torch.std(d))        # standard deviation
print(torch.prod(d))       # product of all numbers

# 단일요소 텐서
agg = tensor.sum()   # 1 + 2 + ... + 12 = 78
agg_item = agg.item()
print(agg_item, type(agg_item))

# 바꿔치기(in-place) 연산
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

#텐서를 NumPy배열로 변환하기
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 텐서의 변경 사항이 NumPy 배열에 반영된다.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy 배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")