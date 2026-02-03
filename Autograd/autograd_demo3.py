import torch

x = torch.tensor(6.7) # Input feature
y = torch.tensor(0.0) # True lavel

w = torch.tensor(1.0, requires_grad=True) # weight
b = torch.tensor(0.0 , requires_grad=True) # Bias


def binary_cross_entropy_loass(prediction , target) :
    epsilon = 1e-8 # to prevent log0
    prediction = torch.clamp(prediction , epsilon , 1-epsilon)
    return -(target * torch.log(prediction) + (1-target) * torch.log(1-prediction))

#forward pass

z = w * x + b # weighted sum

y_pred = torch.sigmoid(z) # predicted proality

#compute inary cross entropy loss

loss = binary_cross_entropy_loass(y_pred,y)

loss.backward()

print(w.grad)
print(b.grad)
    
