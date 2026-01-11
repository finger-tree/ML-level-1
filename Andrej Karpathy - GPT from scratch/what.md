encoding - relation from words to numbers
encode and decode characeters (from string to numbers)
get batcn

head - 
key query value - linear layers of attention

key and query transposed doted 
with each other to form relationship of affect and affected by in terms of change in the embedding

key is the change of embedding to signal what information to give to other time steps

query is the change of embedding to signal what information is to change itself given various tokens at different time positions

value is the change of embedding to reflect given keys and queries dotted, what to do with that information

they can't see the future so we use a tril matrix and softmax to mask querying to the future
we actually softmax with complement of tril masked with -inf, 
this gives the remaining a weighted average

dropout prevents overfitting by randomly shutting down connections during forward and backward pass (graduate programs)


