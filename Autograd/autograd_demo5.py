import torch

x = torch.tensor([1.0,2.0,3.0], requires_grad=True)

y = (x**2).mean()

print(y)

y.backward()

print(x.grad)

# clearing grad

x = torch.tensor(2.0 , requires_grad=True)

y = x ** 2

y.backward()

print(x.grad.zero_())

y = x ** 2

y.backward()

print(x.grad)


# disable gradient tracking

