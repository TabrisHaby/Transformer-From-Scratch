import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """
        Indexing of input vocabulary is transferred to current dimension of embedding.
        
        Args:
            vocab_size (_type_): size of the vocabulary
            d_model (_type_): dimension of the embedding
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        positional embedding for input sequence(# 3.5 in paper)
        odd or even position of the sequence is transferred to the same dimension of embedding by using sin and cos function
        
        Args:
            d_model (_type_): dimension of the embedding
            max_len (_type_): maximum length of the sequence
            device (_type_): CPU or GPU
        """
        super(PositionEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # postion encoding no need gradient
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # 2d tensor
        # index: odd and even
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:,0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:,1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        forward direction

        Args:
            x (_type_): input 
        """
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_p,device):
        """
        combine the token embedding and positional embedding together as transformer embedding

        Args:
            drop_p (_type_): dropout probability, prevent overfitting
        """
        super(TransformerEmbedding, self).__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p = drop_p) # for overfitting
        
    def forward(self, x):
        """
        Trabsformer embedding = dropout(token embedding + positional embedding)
        
        """
        return self.drop_out(self.token_emb(x) + self.pos_emb(x))

       