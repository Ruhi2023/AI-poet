#reading file
with open('Data-input.txt', 'r', encoding='utf-8') as f:
  tx= f.read()
print("length of dataset in characters:",len(tx))
chars = sorted(list(set(tx)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
sti = {ch:i for i,ch in enumerate(chars)}
its = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [sti[c] for c in s]
decode = lambda l: ''.join([its[i] for i in l])
print(encode("hello world"))
print(decode(encode("hello world")))
import torch
data = torch.tensor(encode(tx), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
