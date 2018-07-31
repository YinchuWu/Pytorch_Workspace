import torch
import random


class DynamicNet(torch.nn.Module):
    """docstring for Dynamic_net"""

    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.fc1 = torch.nn.Linear(D_in, H, bias=True)
        self.fc2 = torch.nn.Linear(H, H, bias=True)
        self.fc3 = torch.nn.Linear(H, D_out, bias=True)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        relu = torch.nn.ReLU(inplace=False)
        h_relu = relu(self.fc1(x))
        for i in range(random.randint(0, 5)):
            h_relu = relu(self.fc2(h_relu))
        y_pred = self.fc3(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(model.parameters(
), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

for t in range(500):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    print(t, loss.item())
    optimizer.step(closure=None)
