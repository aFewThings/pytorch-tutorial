import torch

# Tensor
# 만약 .requires_grad 속성을 True 로 설정하면, 그 tensor에서 이뤄진 모든 연산들을 추적(track)하기 시작합니다.
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
print(a.grad_fn) # requires_grad의 입력값이 지정되지 않으면 기본값은 False, 연산을 추적할 수 없다.
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# Gradient
out.backward()
print(x.grad)

x = torch.ones(3, requires_grad=True)
y = x * x
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad) # dy/dx

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
