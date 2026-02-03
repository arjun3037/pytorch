import torch

x = torch.tensor(2.0 , requires_grad=True)

y = x ** 2

print(y)

y.backward()

print(x.grad)
print(x.grad)

x.requires_grad_(False)

y = x ** 2

print(x.grad)
print(y)