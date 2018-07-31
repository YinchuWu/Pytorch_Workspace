import torch
import numpy as np
a = torch.randn(5, 3)
print(a)

b = torch.zeros(5, 3, dtype=torch.int)
print(b)

c = torch.tensor([1, 5.1, 3, 4])
print(c.reshape(2, 2))

x = b.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(b, dtype=torch.double)
print(x)

x = x.view(15)
print(x.shape[0])

if torch.cuda.is_available():
    device = torch.device('cuda')
    b = torch.ones_like(a, device=device)       #set running device
    a = a.to(device)                            #change running device
    c = a + b
    print(c)
    print(c.to('cpu', torch.double))

a = torch.randn([5, 3], requires_grad=True)
print(a)
b = a * 2
b.backward(torch.ones_like(a))  # ??
print(a.grad)
print(b.requires_grad)
with torch.no_grad():
    c = b * 2
    print(c.requires_grad)
a.requires_grad_(False)
print(a.requires_grad)
b_no_grad = b.detach()
print(b_no_grad.requires_grad)
