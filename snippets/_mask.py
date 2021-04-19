import torch

src = torch.arange(9).float().reshape((3, 3))
mask = torch.tensor([[True, True, True],
                     [False, False, True],
                     [True, False, True]])
value = torch.randn(3, 3)
src[mask] = value[mask]
print(src)