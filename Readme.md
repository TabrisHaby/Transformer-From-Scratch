# Project Goal

- The goal of this project is to implement a transformer model from scratch.
- I will use PyTorch for parts of the framework. This project can run on both CPU and GPU (device = device).

# About transformer model

## There are some key parts for transformer model:

**Encoder**
- Input Embedding
- Positional EMbedding
- multi-head attention
- Add & Normal
- Feed Forward

**Decoder**
- Output Embedding
- Positional EMbedding
- **Masked** Multi-Head Attention
- Add & Normal
- Feed Forward

**Classification Head**
- Linear
- Softmax

![Transformer Structure](/pic/image.png)

For Input, there are 2 important information need to embedding: **Input Embedding** and **Positional Embedding**

![Embedding](/pic/image-1.png)

## Embedding

### Input/Output Embedding

- Input: Input Sequence(text, time series)
- Output: Expected Output Sequence

When doing Embedding, usually we use super on torch.nn.Embedding module, there are 3 key parameters:

- num_embeddings: how many characters are embedded as input
- embedding_dim: dimention of output (always named as "d_model")
- padding_idx = None: if i need to make length of all output same

[Source](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

![Embedding_dim](/pic/embedding_dim.png)


### Positional Embedding


![Positional Embedding](/pic/Position_Embedding.png)

- We need to import position information, where pos = positional information and i = dimensition. That means the output of "PE" will be a 2D array: pos and d_model(embedding_dim)

![output_dim](/pic/output_dim.png)

- d_model doesn't need higher if pos is not very long


### Positional Embedding + Input/Output Embedding

Dim of Input/Output Embedding: \[pos, d_model\]
Dim of Positional Embedding: \[d_model, pos\]

Output = dropout(Positional Embedding + Input/Output Embedding^T)

![final_dim](/pic/final_dim.png)


## Multi Head Attention

### Scaled-Dot-Product Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the **query**, **keys**, **values**, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

In the encoder stage, they each carry the same input sequence after this has been embedded and augmented by positional information. Similarly, on the decoder side, the queries, keys, and values fed into the first attention block represent the same target sequence after this would have also been embedded and augmented by positional information. The second attention block of the decoder receives the encoder output in the form of keys and values and the normalized output of the first attention block as the queries. The dimensionality of the queries and keys is denoted by, whereas the dimensionality of the values is denoted by.

![Scaled Dot-Product Attention](/pic/Scaled_Dot_Product_Attention.png)

process: input vector(d_model) -> Linear transformation(torch.nn.Linear) -> q_1; k_1; v_1; -> attention score

- Query: input dim: (seq_length, d_model) -> Y = XW + B -> output dim: (seq_length, dim_q)
- Key: input dim: (seq_length, d_model) -> Y = XW + B -> output dim: (seq_length, dim_k)
- Value: input dim: (seq_length, d_model) -> Y = XW + B -> output dim: (seq_length, dim_v)

- attention score: $score_{11} = q_1*k_1^T$, dim = 1*1 = numeric number; required: dim_q = dim_k

- supposet seq_length = 9, then we have 9 score: $score_{11}, ..., score_{19}$, we want to know the importance of each score z(character), we use softmax method to calculate weight for each score(character).

![attention](/pic/attention.png)

![formula with example](/pic/formula_with_example.png)

![score and softmax](/pic/score_and_softmax.png)

After getting the Softmax matrix, we do a matrix multiple with V matrix to get output as $dim(output) = (pos, dim_v)$


### What is Multi-Head

We obtain the Scaled Dot-Product Attention for a character. When we apply this to all characters, we achieve multi-layer attention, like this:

![multi head](/pic/multi_head.png)

![dim of multi head](/pic/multi_head_dim.png)

Multi-Head Attention projects the same sentence into different dimensional spaces using various linear transformations. This allows the model to learn the similarities between different characters from different perspectives, enhancing the model's performance.

For multi-Head, dim of all layers are as below:

- V: (pos, dim_v) -> (heads, score) -> (heads, pos, dim_v/heads)
- K: (pos, dim_k) -> (heads, score) -> (heads, pos, dim_k/heads)
- Q: (pos, dim_q) -> (heads, score) -> (heads, pos, dim_q/heads)
- Scaled Dot-Product Attention: (heads, pos, pos) -> (heads, score) -> (heads, scaled score) -> (heads, pos, dim_v/heads)
- Concat: (heads, pos, dim_v/heads) -> (pos, dim_v)
- Linear: (pos, dim_v) -> (pos, d_model) | this step is requred for Add & norm


## Add & Norm

### Goal of Add & norm

- Add & Norm are actually two separate steps. The add step is a residual connection, meaning we sum the output of a layer with its input, F(x) + x. This helps address the **vanishing gradient problem**.

- The norm step involves layer normalization. This is one of several computational techniques that facilitate easier model training, improving performance and reducing training time.

![Add & Norm](/pic/Add_Norm.png)

### Formula

- ADD: x + Sublayer(x), where Sublayer = Mut.Head.Attn or Feed Forward Network(FFN)
- Norm: LayerNorm(x + Mut.Head.Attn(x))


## Feed Forward

FFN = Linear Transformation(Relu(Linear transformation))

$FFN(x) = max(0, xW_1+b_1)W_2+b_2$


## Mask

- This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position 𝑖 depend only on the known outputs at positions less than 𝑖. This means the model doesn't need to know the output after 𝑖 + 2.

- pad_mask : mask in padding processing
- sub_mask : mask in decoding processing

$trg sub mask = torch.tril(torch.ones(trg len, trg len)).type(torch.Bytetensor).to(self.device)$
$trg_mask = trg_pad_mask & trg_sub_mask$