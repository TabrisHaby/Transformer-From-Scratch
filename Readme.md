## Project Goal

- The Goal of this project is to implement a transformer model from scratch.
- I will use PyTorch for some of the framework, this project can be run on both CPU and GPU(device = device)

## About transformer model

- There are some key parts for transformer model:

1. Embedding
2. Multi-Head Attention
3. Encoder
4. Decoder
5. Layer Norm
6. Corss Attention

### Embedding

- Token Embedding : index of input vacubulary convert into certain dim of embedding
- Position Embedding: convert for every word, position encoding by using sin/cos 
- Transformer Embedding : tensor of transformer = dropout(token embedding + position embedding)