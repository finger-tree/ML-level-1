import torch
import torch.nn as nn
embedding = nn.Embedding(4, 1)
input_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
output = embedding(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Embedding:", output)
print("Output Shape:", output.shape)