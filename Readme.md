### Project Goal

- The Goal of this project is to implement a transformer model from scratch.
- I will use PyTorch for some of the framework, this project can be run on both CPU and GPU(device = device)

## About transformer model

- There are some key parts for transformer model:

    - Embedding
    - Multi-Head Attention
    - Encoder
    - Decoder
    - Layer Norm
    - Corss Attention

### Embedding

- Token Embedding : index of input vacubulary convert into certain dim of embedding
- Position Embedding: convert for every word, position encoding by using sin/cos 
- Transformer Embedding : tensor of transformer = dropout(token embedding + position embedding)


### Multi Head Attention

- The goal of Multi-Head Attention is to allow the model to focus on different parts of the input sequence for each attention head, thereby capturing various aspects of the information.

- In the context of Transformer models, which are widely used in natural language processing tasks, the attention mechanism is used to weigh the relevance of different words in the input sequence when generating each word in the output sequence. However, using a single attention head might limit the ability of the model to capture different types of dependencies between words.

- By using multiple attention heads, each with its own learned linear transformations (for Query, Key, and Value), the model can learn to pay attention to different types of information. For example, one head might learn to pay attention to syntactic information (like subject-verb agreement), while another might focus on semantic information (like word meaning and context).

- So, the main goal of Multi-Head Attention is to enrich the ability of the attention mechanism to capture various types of dependencies in the data, thereby improving the performance of the model on complex tasks like machine translation, text summarization, etc. It’s like having multiple perspectives to understand the data better.

### Structure of Transformer

#### Encoding

- Word Embedding
- Position Embedding
- Encoder layer: Multi Head; PoswiseFeedForward

### Decoding

- Word Embedding
- Position Embedding
- Decoder layer: Multi Head; Multi Head; PoswiseFeedForward