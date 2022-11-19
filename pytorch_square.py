import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Creating the basick modell

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack =  nn.Sequential(
         nn.Linear(1,15),
         nn.ReLU(),
         nn.Linear(15,15),
         nn.ReLU(),
         nn.Linear(15,15),
         nn.ReLU(),
         nn.Linear(15,1),
         )
    def forward(self,x):
        logits = self.linear_stack(x)
        return logits

#print(model(torch.rand(10,1)))


model = NeuralNetwork()

#use optimizer which will deal with bias and gradients 

#train_set
batch_size = 10000
X = torch.rand(batch_size,1)
Y = X**2

lr = 0.01
optim = torch.optim.Adam(model.parameters(), lr=lr)

loss = torch.nn.MSELoss()
n_iters = 2000


for epochs in range(n_iters):
    #forward pass and loss
    out = model(X)
    l = loss(out,Y)
    #backward pass
    l.backward()
    #update weights
    optim.step()
    optim.zero_grad()

    if epochs % 100 == 0:
        print(f" {epochs+1}, l={l.item()}")

X_test = torch.rand(10,1)
print(model.parameters)
print(f"Real outcome : {X_test**2}")
print(f"Predicted output : {model(X_test)}")


