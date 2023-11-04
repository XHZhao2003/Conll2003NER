import torch

a = torch.Tensor([[1, 2, 3]])
b = torch.Tensor([[3, 2, 1]])
c = [a, b]
d = torch.cat([x for x in c], 0)
print(d)