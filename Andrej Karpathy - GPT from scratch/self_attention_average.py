"""
self attention using the sum of past embedded token values:
a simple way past information is relayed to future tokens
"""


import torch
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape

# three ways to compute the average of all previous tokens up and including t
# first we loop through each of the batch and time dimensions and add them up
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1, :]                   # for each b, get all up to t in T
        xbow[b, t, :] = torch.sum(xprev, 0) / (t+1)   # save for each t of b the sum of all previous t tokens

# so it through broadcasting a division of sum in a dimension to a triangular matrix
# matmul by triangular one matrix analogous to [x[b, :t+1, :] for t in range(T)] for t in b
# then normalize (so rows sum to 1) so that matmul is the average
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (T,T) @ (B,T,C) --> (B,T,C) via broadcasting
print(torch.allclose(xbow, xbow2))

# use softmax 
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3))
