import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        """_summary_
        multi-headed attention

        Args:
            n_head (_type_): count of head
            d_model (_type_): dimension of model

        """
        super(MultiHeadAttention, self).__init__()
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


def get_attn_pad_mask(seq_q, seq_k):
    """
    For masking out the padding part of the sequence
    letting the model know which is the padding part that no need to be attended
    so that padding part can be set to inf before softmax

    Args:
        seq_q (_type_): query sequence, which is the output of the encoder
        seq_k (_type_): key sequence, which is the input of the decoder

    Returns:
        pad_attn_mask: [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    """
    For masking out the subsequent info

    Args:
        seq (_type_): _description_

    Returns:
        subsequent_mask: [batch_size, len, len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    subsequent_mask = subsequent_mask.bool()
    return subsequent_mask

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w1(x)))

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v, d_v]
        # attn_mask: [batch_size, seq_len, seq_len]

        # mask position will be set to -inf so that softmax(x) = 0
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_model)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # even position
        pe[:, 1::2] = torch.cos(position * div_term) # odd position

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        Args:
            enc_inputs: [batch_size, src_len, d_model]
            enc_self_attn_mask: [batch_size, src_len, src_len]

        Returns:
            enc_outputs: [batch_size, src_len, d_model]
            enc_self_attns: [batch_size, n_heads, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attns: [batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, enc_self_attns

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, n_head, d_ff):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_input):
        """_summary_

        Args:
            enc_input (_type_): Encoder input, shape: [batch_size, src_len]

        Returns:
            enc_outputs: [batch_size, src_len, d_model]
        """
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_input)
        # positional encoding
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # enc_self_attn_mask: for padding information, shape: [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_input, enc_input)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, dec_input, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_input, dec_input, dec_input, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers, n_head, d_ff):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(n_layers))

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # dec_inputs: [batch_size, tgt_len]
        dec_outputs = self.tgt_emb(dec_inputs)
        # positional encoding
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)
        # dec_self_attn_pad_mask: for masking out the padding part of the sequence
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # dec_self_attn_subsequent_mask: for masking out the subsequent info
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # dec_self_attn_subsequent_mask: [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequent_mask),0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attns)

        return dec_outputs, dec_self_attns, dec_enc_attns



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_input, dec_input):
        """_summary_

        transformer model forward direction

        Args:
            enc_input (_type_): Encoder input, shape: [batch_size, src_len]
            dec_input (_type_): Decoder input, shape: [batch_size, tgt_len]

        Returns:
            dec_logits.view(-1, dec_logits.size(-1)): [batch_size * tgt_len, tgt_vocab_size]
            enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
            dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
            dec_enc_attns: [n_layers, batch_size, n_heads, tgt_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_input)
        # dec_outpus: [batch_size, tgt_len, d_model]
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attn: [n_layers, batch_size, n_heads, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_input, enc_input, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
    

if __name__ == "__main__":
    # Define model parameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    n_layers = 6
    n_head = 8
    d_ff = 2048

    # Instantiate the model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_head, d_ff)

    # Define sample inputs
    batch_size = 32
    src_len = 10
    tgt_len = 10

    enc_input = torch.randint(0, src_vocab_size, (batch_size, src_len)).cuda()
    dec_input = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len)).cuda()

    # Forward pass
    dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_input, dec_input)

    # Print the shapes of the outputs to verify correctness
    print(f"dec_logits shape: {dec_logits.shape}")
    print(f"enc_self_attns length: {len(enc_self_attns)}")
    print(f"dec_self_attns length: {len(dec_self_attns)}")
    print(f"dec_enc_attns length: {len(dec_enc_attns)}")