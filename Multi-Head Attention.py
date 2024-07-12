import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

class MutliHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        """_summary_
        multi-headed attention

        Args:
            n_head (_type_): count of head
            d_model (_type_): dimension of model

        """
        super(MutliHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model

        # Q, K, V = Query, Key, Value
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # combine the Q, K, V
        self.W_O = nn.Linear(d_model, d_model)

        # active function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """_summary_
        multi-headed attention: forward direction, calculate tensor

        Args:
            Q (Tensor): (batch_size, seq_len_q, d_model)
            K (Tensor): (batch_size, seq_len_k, d_model)
            V (Tensor): (batch_size, seq_len_v, d_model)
            mask:prevent attention to see something shouldn't been seen, like padding
        """
        # x = torch.rand(batch = 128, seq_len = 64, d_model = 512)
        batch, time, dim = Q.shape
        n_dim = self.d_model // self.n_head
        # linear transformation for Q, K, V
        Q,K,V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        # rebuild Q,K,V as multi-head.shape
        Q = Q.view(batch, time, self.n_head, n_dim).permute(0,2,1,3)
        K = K.view(batch, time, self.n_head, n_dim).permute(0,2,1,3)
        V = V.view(batch, time, self.n_head, n_dim).permute(0,2,1,3)
        # calculate the multi-head attention score
        score = torch.matmul(Q, K.transpose(2,3)) / math.sqrt(n_dim)
        # mask: if mask is not None, then make sure score of attention after softmax is 0
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        # active function
        score = self.softmax(score)
        # combine the Q, K, V
        score = torch.matmul(score, V)
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dim)

        out = self.W_O(score)
        return out
    

def main():
    x = torch.rand(batch = 128, seq_len = 64, d_model = 512)
    model = MutliHeadAttention(n_head = 8, d_model = 512)
    out = model(x, x, x)
    print(out.shape)


if __name__ == "__main__":
    main()
