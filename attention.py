import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.module() and nn.Linear()
import torch.nn.functional as F # This gives us the softmax()

class Attention(nn.Module):
    def __init__(self, d_k, d_v, d_model=2,
                 row_dim=0,
                 col_dim=1):
        """
        Calculates the attention scores for the given query, key and value vectors

        input:
           d_k: dimension of query and key vectors (its important that they have same dimension due to the compatibility of the dot product)
           d_v: dimension of value vector
           d_model: dimension of input vectors of model
           row_dim: axis for rows
           col_dim: axis for columns
        """

        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_k, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_k, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_v, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim


    # The only change from SelfAttention and attention is that
    # we expect 3 sets of encodings to be passed in...
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        # ...and we pass those sets of encodings to the various weight matrices.
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        return attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v,
                 d_model=2,
                 row_dim=0,
                 col_dim=1,
                 num_heads=1):
        """
        Calculates multiple attention heads for the given query, key and value vectors

        input:
           d_k: dimension of query and key vectors (its important that they have same dimension due to the compatibility of the dot product)
           d_v: dimension of value vector
           d_model: dimension of input and output vectors of model
        """

        super().__init__()

        ## create a bunch of attention heads
        self.heads = nn.ModuleList(
            [Attention(d_k, d_v, d_model, row_dim, col_dim)
             for _ in range(num_heads)]
        )

        # We want to make sure the output has dimension d_model
        self.out = nn.Linear(in_features=num_heads*d_v,
                             out_features=d_model,
                             bias=False)

        self.col_dim = col_dim

    def forward(self,
                encodings_for_q,
                encodings_for_k,
                encodings_for_v,
                mask=None):

        ## run the data through all of the attention heads
        return self.out(torch.cat([head(encodings_for_q,
                               encodings_for_k,
                               encodings_for_v,
                               mask)
                          for head in self.heads], dim=self.col_dim))



class EncoderLayer(nn.Module):
  def __init__(self, d_k=64, d_v=64, d_model=512, row_dim=0, col_dim=1, num_heads=8, p_drop=0.1):

    """
    This creates a single encoder layer, with multi-head attention and feed forward layers.
    """

    super().__init__()

    self.multihead = MultiHeadAttention(d_k, d_v, d_model, row_dim, col_dim, num_heads)
    self.norm1 = nn.LayerNorm(d_model)
    self.ff1 = nn.Linear(in_features = d_model, out_features = d_model, bias = True)
    self.relu = nn.ReLU()
    self.ff2 = nn.Linear(in_features=d_model, out_features = d_model, bias = False)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=p_drop)

  def forward(self, encodings, mask = None):
    x = self.multihead(encodings, encodings, encodings, mask)
    x = self.norm1(x + encodings)
    x = self.dropout(x)

    y = self.ff1(x)
    y = self.relu(y)
    y = self.ff2(y)
    y = self.norm2(y + x)
    y = self.dropout(y)

    return y



class Encoder(nn.Module):
  def __init__(self, d_k=64, d_v=64, d_model=512, row_dim=0, col_dim=1, num_heads=8, p_drop=0.1, N = 6):
    """
    This repeats the encoder layer N times.
    """
    super().__init__()

    self.layers = nn.ModuleList([EncoderLayer(d_k, d_v, d_model, row_dim, col_dim, num_heads, p_drop)
                                 for _ in range(N)])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, encodings, mask = None):
    for layer in self.layers:
      encodings = layer(encodings, mask)

    return self.norm(encodings)




class DecoderLayer(nn.Module):
  def __init__(self, d_k=64, d_v=64, d_model=512, row_dim=0, col_dim=1, num_heads=8, p_drop=0.1):

    super().__init__()

    self.masked_multihead = MultiHeadAttention(d_k, d_v, d_model, row_dim, col_dim, num_heads)
    self.norm1 = nn.LayerNorm(d_model)
    self.multihead = MultiHeadAttention(d_k, d_v, d_model, row_dim, col_dim, num_heads)
    self.norm2 = nn.LayerNorm(d_model)
    self.ff1 = nn.Linear(in_features = d_model, out_features=d_model, bias = True)
    self.relu = nn.ReLU()
    self.ff2 = nn.Linear(in_features=d_model, out_features=d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=p_drop)
    self.row_dim = row_dim
    self.col_dim = col_dim
    self.num_heads = num_heads

  def forward(self, encoder_encodings, decoder_encodings, mask = None):
    x = self.masked_multihead(decoder_encodings, decoder_encodings, decoder_encodings, mask)
    x = self.norm1(x + decoder_encodings)
    x = self.dropout(x)

    y = self.multihead(x, encoder_encodings, encoder_encodings, None)
    y = self.norm2(y + x)
    y = self.dropout(y)

    z = self.ff1(y)
    z = self.relu(z)
    z = self.ff2(z)
    z = self.norm3(z + y)
    z = self.dropout(z)

    return z



class Decoder(nn.Module):
  def __init__(self, d_k=64, d_v=64, d_model=512, row_dim=0, col_dim=1, num_heads=8, p_drop=0.1, N = 6):

    super().__init__()

    self.layers = nn.ModuleList([DecoderLayer(d_k, d_v, d_model, row_dim, col_dim, num_heads, p_drop)
                                 for _ in range(N)])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, encoder_encodings, decoder_encodings, mask = None):
    for layer in self.layers:
      decoder_encodings = layer(encoder_encodings, decoder_encodings, mask)

    return self.norm(decoder_encodings)



def pos_encodings(token_len, d_model):
  pos_enc = torch.zeros((token_len, d_model), dtype = torch.float32)
  for pos in range(token_len):
    for i in range(d_model):
      if i % 2 == 0:
        pos_enc[pos, i] = torch.sin(torch.tensor(pos / (10000 ** (i / d_model))))
      else:
        pos_enc[pos, i] = torch.cos(torch.tensor(pos / (10000 ** ((i - 1) / d_model))))

  return pos_enc


