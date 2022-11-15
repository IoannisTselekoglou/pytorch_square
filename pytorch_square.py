import torch
import torch.nn as nn


#Creating the basick modell

model = nn.Sequential(
  nn.Linear(1,10),
  nn.ReLU(),
  nn.Linear(10,10),
  nn.ReLU(),
  nn.Linear(10,10),
  nn.ReLU(),
  nn.Linear(10,10),
  nn.ReLU(),
  nn.Linear(10,10),
  nn.ReLU(),
  nn.Linear(10,1)
)




#print(model(torch.rand(10,1)))


a = sum(model(torch.rand(10,1)))


#use optimizer which will deal with bias and gradients 

batch_size = 10000

X = torch.rand(batch_size,1)
Y = X**2

optim = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 1000
for i in range(epochs):
  out = model(X)
  loss = sum((out-Y)**2/batch_size)
  print(loss)
  loss.backward()
  optim.step() #diesen schritt noch besser versethen 
  optim.zero_grad() #diesen schritt noch besser versethen 



X = torch.rand(2,1)
print(X**2)
print(model(X))
