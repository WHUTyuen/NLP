import torch
from  torch import nn
a = nn.Embedding(10,3)
print(a.weight)
idx = torch.tensor([[1,3,4,5],[1,3,4,5]])
print(a(idx))