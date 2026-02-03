import torch

# if we need derivatives of this tensor in future , we need to mark it requires_grad = True
x = torch.tensor(3.0 , requires_grad = True)

y =  x**2;

print(x)
print(y)

# calculate dy/dx 
y.backward()

print(x.grad)