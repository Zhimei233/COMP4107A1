# Name this file assignment1.py when you submit
import torch

# A function simulating an artificial neuron
def artificial_neuron(x, w):
  # x is a list of inputs of length n
  # w is a list of inputs of length n
  
  # convert lists to torch tensors
  x = torch.tensor(x, dtype=torch.float32)
  w = torch.tensor(w, dtype=torch.float32)

  weighted_sum = torch.sum(x * w)
  # apply the activation function SiLU
  op = weighted_sum / (1 + torch.exp(-weighted_sum))
  
  output = op.item()
  return output


# A function performing gradient descent
def gradient_descent(f, df, x0, alpha):
  # f is a function that takes as input a list of length n
  # df is the gradient of f; it is a function that takes as input a list of length n
  # x0 is an initial guess for the input minimizing f
  # alpha is the learning rate
  argmin_f = x0.copy()
  f_old = f(argmin_f)
  max_iters = 1000
  
  for iteration in range(max_iters):
    gradient = df(argmin_f)
    
    # update x using the gradient and learning rate
    for i in range(len(argmin_f)):
      argmin_f[i] = argmin_f[i] - alpha * gradient[i]
    
    f_new = f(argmin_f)
    
    # check for convergence
    if abs(f_new - f_old) < 1e-6:
      break
      
    f_old = f_new
  
  min_f = f_new

  # argmin_f is the input minimizing f
  # min_f is the value of f at its minimum
  return argmin_f, min_f


# A function that returns a neural network module in PyTorch
def pytorch_module():
  
  class SimpleNet(torch.nn.Module):
    def __init__(self):
      super(SimpleNet, self).__init__()
      self.linear1 = torch.nn.Linear(10, 5)  # input layer to hidden layer
      self.linear2 = torch.nn.Linear(5, 1)   # hidden layer to output layer
      self.activation = torch.nn.ReLU()   # activation function

    def forward(self, x):
      x = self.linear1(x)
      x = self.activation(x)
      x = self.linear2(x)
      return x
  module = SimpleNet()
  # A pytorch module
  return module