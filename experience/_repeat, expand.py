import torch

x = torch.randn((1, 5, 5))
print(x)

x = x.unsqueeze(1)
print(x)
# expand와 repeat은 특정 차원을 복제해서 반복시킨다. 
x = x.expand(-1, 2, -1, -1) # expand는 데이터 복사가 없음
#x = x.repeat(1, 2, 1, 1) repeat은 데이터를 복사함
print(x)