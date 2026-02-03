import torch

x = torch.tensor(4.0 , requires_grad = True)

y =  x**2;

z = torch.sin(y)

print(x)
print(y)
print(z)

z.backward()

print(x.grad)