"""Top-level model classes.
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
import util

from typing import List, Tuple, Dict, Set, Union

class Seq2Seq(nn.Module):
    """Baseline seq2seq model
    Implements a basic seq2seq network (without attention):
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Decode layer: Decode the encoded sequence word by word.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        output_size(int): Number of logits for softmax layer
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, output_size, drop_prob=0.):
        super(Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.word_vectors = word_vectors
        
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.encoder = layers.EncoderRNN(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.decoder = layers.DecoderRNN(input_size=hidden_size, 
                                        hidden_size=hidden_size,
                                        num_layers=1)

        self.projection = nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, cw_idxs, qw_idxs):
        batch_size = cw_idxs.size(0)
        
        # Chop of the EOS token.
        qw_idxs = qw_idxs[:, :-1]

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc, dec_init_state = self.encoder(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        
        decoder_outputs = []
        h_0, c_0 = dec_init_state        #(batch_size, hidden_size)
        h_0 = h_0.contiguous().view(batch_size, 1, self.hidden_size)  # Assuming layer dimension is 1
        c_0 = c_0.contiguous().view(batch_size, 1, self.hidden_size)  # Assuming layer dimension is 1
        decoder_hidden = (h_0, c_0)  

        for q_t in torch.split(q_emb, split_size_or_sections=1, dim=1):         #(batch_size, 1, hidden_size)
            o_t, decoder_hidden = self.decoder(q_t, decoder_hidden)
            decoder_outputs.append(o_t)
    
        decoder_outputs = torch.stack(decoder_outputs, dim=1).squeeze(dim=2)       #(batch_size, q_len, hidden_size)        
        logits = self.projection(decoder_outputs)                   #(batch_size, q_len, output_size)        
        
        # Mask not needed, as we can simply ignore pad tokens in the loss function
        #q_mask.unsqueeze_(-1)
        #q_mask = q_mask.expand(logits.size(0), logits.size(1), logits.size(2))        
        #log_probs = util.masked_softmax(logits, q_mask, dim=-1, log_softmax=True)       #(batch_size, q_len, output_size)
        
        log_probs = F.log_softmax(logits, dim=-1)        #(batch_size, q_len, output_size)

        return log_probs

    def step(self, qw_idx_t: torch.Tensor,
            decoder_init_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the RNN decoder

        @param qw_idx_t (Tensor): t_th word Index of question.
                Shape (batch_size, 1).
        @param decoder_init_state (Tuple(Tensor)): Decoder's prev hidden state
                Shape (batch_size, 1, hidden_size)

        @returns dec_hidden (Tensor): Decoder's hidden state after passing q_t and previous hidden state
                Shape (batch_size, hidden_size)
        @returns log_probs (Tensor): Soft prediction for next word index
                Shape ((batch_size, 1, output_size))
        """

        q_t = self.emb(qw_idx_t)   # (batch_size, 1, hidden_size)
        decoder_hidden = decoder_init_state #(batch_size, 1, hidden_size)  

        o_t, decoder_hidden = self.decoder(q_t, decoder_hidden)
        logits = self.projection(o_t)   #(batch_size, 1, output_size)
        log_probs = F.log_softmax(logits, dim=2)    #(batch_size, 1, output_size)
        return decoder_hidden, log_probs