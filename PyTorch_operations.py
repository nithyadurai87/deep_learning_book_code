import torch

x = torch.Tensor([[110,120,90],[160,20,60]])

print (x.topk(k = 1, dim = 1))
print (x.topk(k = 2, dim = 1))
print (x.topk(k = 1, dim = 0))
print (x.topk(k = 2, dim = 0))

print (torch.Tensor([110]).item())
print (x.mul_(2))

y = x.numpy()
z = torch.from_numpy(y)
print (type(y))
print (type(z))

a = x.view(1,-1)
print(a)
print(a.size())