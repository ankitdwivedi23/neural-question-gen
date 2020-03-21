"""Assortment of layers for use in models.py.

code adapted from:
    > https://github.com/chrischute/squad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Word-level Embedding layer.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        #emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        #emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class EncoderRNN(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 drop_prob=0.):
        super(EncoderRNN, self).__init__()
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)
        
        self.h_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)
        self.c_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)
        batch_size = x.size(0)

        # Sort by length and pack sequence for RNN
        #print("Before sorting:")
        #print(x)
        #lengths, sort_idx = lengths.sort(0, descending=True)
        #print("Lengths shape:")
        #print(lengths.shape)
        #x = x[sort_idx]     # (batch_size, seq_len, input_size)
        #print("Sorted:")
        #print(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        #print("Packed Sequence")
        #print(x)

        # Flatten RNN params
        self.rnn.flatten_parameters()

        # Apply RNN
        x, (last_hidden, last_cell) = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        #_, unsort_idx = sort_idx.sort(0)
        #x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        #enc_hiddens = F.dropout(x, self.drop_prob, self.training)

        #Concatenate last hidden state of last encoder layer
        last_hidden = last_hidden.contiguous().view(self.rnn.num_layers, 2, batch_size, self.rnn.hidden_size)  # (num_layers, num_directions=2, batch_size, hidden_size)
        last_hidden = torch.cat((last_hidden[:,0:1,:,:], last_hidden[:,1:2,:,:]), dim=-1).squeeze(1)  # (num_layers, batch_size, 2 * hidden_size)
        last_cell = last_cell.contiguous().view(self.rnn.num_layers, 2, batch_size, self.rnn.hidden_size)  # (num_layers, num_directions=2, batch_size, hidden_size)
        last_cell = torch.cat((last_cell[:,0:1,:,:], last_cell[:,1:2,:,:]), dim=-1).squeeze(1) # (num_layers, batch_size, 2 * hidden_size)

        #Project last hidden and cell state to get initial decoder hidden and cell state
        dec_init_hidden = self.h_projection(last_hidden)
        dec_init_cell = self.c_projection(last_cell)
        dec_init_state = (dec_init_hidden, dec_init_cell)
        #dec_init_state = (last_hidden, last_cell)

        return x, dec_init_state

class DecoderRNN(nn.Module):
    """General-purpose layer for decoding the output of an encoder using RNN.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob)

    def forward(self, input, hidden):
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(input, hidden)
        return output, hidden

class DecoderRNNCell(nn.Module):
    """General-purpose layer for decoding the output of an encoder using RNN.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
    """
    def __init__(self, input_size, hidden_size):
        super(DecoderRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnn(input, hidden)
        return hidden

class Generator(nn.Module):
    """Define standard linear + softmax generation step.
    Args:
        d_model (int): Input dimension
        vocab_size (int): Vocab size (output dimension)
    """
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

#################################################################################################################

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class EncoderSimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderSimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, input, lengths):
        output, hidden = self.rnn(input)
        return output, hidden

class EncoderRNNCell(nn.Module):
    """General-purpose layer for decoding the output of an encoder using RNN.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
    """
    def __init__(self, input_size, output_size, hidden_size, device):
        super(EncoderRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.rnn = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden):
        input = self.embedding(input)
        output, hidden = self.rnn(input, hidden)
        return hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

class DecoderSimpleRNN(nn.Module):
    """General-purpose layer for decoding the output of an encoder using RNN.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
    """
    def __init__(self, input_size, output_size, hidden_size, device):
        super(DecoderSimpleRNN, self).__init__()
        self.input_size = input_size
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.rnn = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden):
        input = self.embedding(input)
        input = F.relu(input)
        output, hidden = self.rnn(input, hidden)
        return output, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)