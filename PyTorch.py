import torch

x1 = torch.Tensor()
print (x1)

x2 = torch.Tensor([1,2,3])
print (x2)
print (x2.size())
print (x2.dtype)

x3 = torch.tensor([1,2,3])
x4 = torch.as_tensor([1,2,3])
print (x3)
print (x4)
print (x3+x4)
#print (x2+x3)

print (x4.dtype)
print (x4.device)
print (x4.layout)

print (torch.eye(2))
print (torch.zeros(2,2))
print (torch.ones(2,2)) 	
print (torch.rand(2))
print (torch.rand(2,3))
print (torch.rand(2,3,4))
