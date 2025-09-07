import torch

device = torch.device("cuda")
x = torch.randn(50000, 50000, device=device)

while True:
    x = torch.matmul(x, x)  # keeps GPU busy

