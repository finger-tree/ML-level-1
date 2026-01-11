import torch
import torch.nn as nn
import torch.nn.functional as F
# a basic linear layer from matrices


torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
head_size = 16

# how linear layers work
# ll_w = torch.randn(head_size, C)                    # linear layers are just layer matrix @ input vector but transposed because vectors are rows
# ll = lambda x: x @ ll_w.T                           # bectors are rows, hence this transposed form
# print(ll(x).shape)
# print(ll_w.shape)

# self attention head in order

key = nn.Linear(C, head_size, bias=False)           
query = nn.Linear(C, head_size, bias=False)         
value = nn.Linear(C, head_size, bias=False)     # input shape (B,T,C) output shape (B,T,head_size), 

# (B,T,C) @ (C,head_size) --> (B,T,head_size)
k = key(x)      #key is the learned projection of a token to other tokens in a sequence
q = query(x)    #query is the learned projection of a token to query other tokens in a sequence
v = value(x)    #value is the learned projection of a token to be aggregated from other tokens in a sequence
wei = q @ k.transpose(-2, -1)                      # q dotted with every k to get attention of learned q and k values from x
tril = torch.tril(torch.ones(T, T))                
wei = wei.masked_fill(tril == 0, float('-inf'))    # mask future tokens (so it can't 'see' them)
wei = F.softmax(wei, dim=-1)                       # softmax to get weights    
out = wei @ v

# note attention is more general, e.g. in sentiment analysis, we can omit the masking and look at future tokens
# k, q, v can also be more general, but not generating from juse the same x input
# in the case of cross_attention, k and v come from different source, and resembles encoder-decoder attention in transformers


class Head(nn.Module):
    def __init__(self, C, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5               # this makes the softmax not peaky, prevents serious bias to higher values
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

