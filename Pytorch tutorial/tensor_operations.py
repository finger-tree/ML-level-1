"""tensor operations in PyTorch"""

import torch
import numpy as np


# Initialize a tensor using array, numpy array
arr = [1, 2, 3]
np_arr = np.array([4, 5, 6])
t1 = torch.tensor(arr)
t2 = torch.from_numpy(np_arr)
print(t1, t2)

# Initialize a tensor using the shape and datatype of another tensor
for i in [torch.zeros_like(t1), torch.ones_like(t1), torch.rand_like(t1, dtype=torch.float)]:
    print(i)

# Initialize a tensor with a shape tuple with 1s 0s randoms
shape = (2, 3,)
for i in [torch.zeros(shape), torch.ones(shape), torch.rand(shape)]:
    print(i)

# Access the shape, dtype, device attributes of a tensor
print("Shape of t1:", t1.shape)
print("Dtype of t1:", t1.dtype)
print("Device of t1:", t1.device)

# Move tensors to torch.accelerator (GPU) using condition torch.accelrator.is_available() and tensor.to
if torch.accelerator.is_available():
    t1 = t1.to(torch.accelerator.current_accelerator())

# Use numpy like indexing and slicing on tensors
t1 = torch.rand(3, 3, 3)
print(t1)
print(t1[0, 0:2, 0:2]) 

# Joining tensors using torch.cat(list, dim=1)
t2 = torch.ones(3, 2, 3)
t3 = torch.cat([t1, t2], dim=1) # dim 1 is t1 dim1 + t2 dim1, other dims should match
print(t3)

