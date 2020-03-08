"""Top-level model classes.

code adapted from:
    > https://github.com/chrischute/squad
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
        - Decoder layer: Decode the encoded sequence word by word.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        output_size(int): Number of logits for softmax layer (vocab size)
        device (string): 'cuda:0' or 'cpu'
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, output_size, device, drop_prob=0., num_layers=1):
        super(Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.word_vectors = word_vectors

        self.emb = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size, padding_idx=0)
        
        self.encoder = layers.EncoderRNN(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     drop_prob=drop_prob)

        self.decoder = layers.DecoderRNN(input_size=hidden_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers)      

        self.projection = nn.Linear(in_features=hidden_size, out_features=output_size)


    def forward(self, cw_idxs, qw_idxs):
        # Chop of the EOS token.
        qw_idxs = qw_idxs[:, :-1]      
        _, dec_init_state = self.encode(cw_idxs)
        _, log_probs = self.decode(dec_init_state, qw_idxs)         #(batch_size, q_len, output_size)
        return log_probs


    def encode(self, cw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        c_len = c_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)

        enc_hiddens, dec_init_state = self.encoder(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        return enc_hiddens, dec_init_state


    def decode(self, dec_init_state: Tuple[torch.Tensor, torch.Tensor], qw_idxs: torch.Tensor) -> torch.Tensor:
        """Compute combined decoder output vectors for a batch.

        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param qw_idxs (Tensor): padded question indices (b, q_len), where
                                       b = batch size, q_len = maximum question length 

        @returns dec_state (tuple(Tensor, Tensor), log_probs (Tensor):   tuple of final decoder hidden state and cell state of shape (b, h), 
                                                                        and log_probs of shape (b, q_len, h), where b = batch_size,
                                                                        q_len = maximum question length,  h = hidden size
        """        
        dec_state  = dec_init_state        #(num_layers, batch_size, hidden_size)

        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        # Initialize a list we will use to collect the combined decoder output o_t on each step
        combined_decoder_outputs = []

        for q_t in torch.split(q_emb, split_size_or_sections=1, dim=1):         #(batch_size, 1, hidden_size)
            o_t, dec_state = self.decoder(q_t, dec_state)                       #(batch_size, 1, hidden_size)
            o_t = o_t.squeeze(1)                                                #(batch_size, hidden_size)
            combined_decoder_outputs.append(o_t)
    
        combined_decoder_outputs = torch.stack(combined_decoder_outputs, dim=1)       #(batch_size, q_len, hidden_size)
        logits = self.projection(combined_decoder_outputs)                   #(batch_size, q_len, output_size)
        log_probs = F.log_softmax(logits, dim=-1)        #(batch_size, q_len, output_size)
        return dec_state, log_probs

    ########################################## Old baseline functions ######################################################    

    def __init__1(self, word_vectors, hidden_size, output_size, drop_prob=0.):
        super(Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.word_vectors = word_vectors
        
        #self.emb = layers.Embedding(word_vectors=word_vectors,
        #                            hidden_size=hidden_size,
        #                            drop_prob=drop_prob)

        self.emb = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        self.encoder = layers.EncoderRNN(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.decoder = layers.DecoderRNN(input_size=hidden_size, 
                                        hidden_size=hidden_size,
                                        num_layers=1)

        self.projection = nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward1(self, cw_idxs, qw_idxs):
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
        #h_0 = h_0.contiguous().view(1, batch_size, self.hidden_size)  # Assuming layer dimension is 1
        #c_0 = c_0.contiguous().view(1, batch_size, self.hidden_size)  # Assuming layer dimension is 1
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
    
    def step1(self, qw_idx_t: torch.Tensor,
            decoder_init_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the RNN decoder

        @param qw_idx_t (Tensor): t_th word Index of question.
                Shape (batch_size, 1).
        @param decoder_init_state (Tuple(Tensor)): Decoder's prev hidden state
                Shape (1, batch_size, hidden_size)

        @returns dec_hidden (Tensor): Decoder's hidden state after passing q_t and previous hidden state
                Shape (batch_size, hidden_size)
        @returns log_probs (Tensor): Soft prediction for next word index
                Shape ((1, batch_size, output_size))
        """

        q_t = self.emb(qw_idx_t)   # (batch_size, 1, hidden_size)
        decoder_hidden = decoder_init_state #(batch_size, 1, hidden_size)  

        o_t, decoder_hidden = self.decoder(q_t, decoder_hidden)
        logits = self.projection(o_t)   #(batch_size, 1, output_size)
        log_probs = F.log_softmax(logits, dim=2)    #(batch_size, 1, output_size)

        return decoder_hidden, log_probs
        

##################################################################################################################

class Seq2SeqAttn(nn.Module):
    """Seq2seq model with attention
    Implements a basic seq2seq network (with attention):
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Decode layer: Decode the encoded sequence word by word.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        output_size(int): Number of logits for softmax layer
        device (string): 'cuda:0' or cpu
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, output_size, device, drop_prob=0., num_layers=1):
        super(Seq2SeqAttn, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.word_vectors = word_vectors
        self.device = device
        self.enc_hiddens = None
        self.enc_masks = None
        
        self.emb = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size, padding_idx=0)

        self.encoder = layers.EncoderRNN(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     drop_prob=drop_prob)

        self.decoder = layers.DecoderRNN(input_size=2*hidden_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers)

        self.att_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(in_features=3*hidden_size, out_features=hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        self.dropout = nn.Dropout(p=drop_prob)
    
    
    def forward(self, cw_idxs, qw_idxs):
        # Chop of the EOS token.
        qw_idxs = qw_idxs[:, :-1]      
        _, dec_init_state = self.encode(cw_idxs)
        _, log_probs = self.decode(dec_init_state, qw_idxs)         #(batch_size, q_len, output_size)
        return log_probs

    
    def encode(self, cw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        c_len = c_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)

        enc_hiddens, dec_init_state = self.encoder(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        
        self.enc_hiddens = enc_hiddens
        self.enc_masks = c_mask

        return enc_hiddens, dec_init_state

    def decode(self, dec_init_state: Tuple[torch.Tensor, torch.Tensor], qw_idxs: torch.Tensor) -> torch.Tensor:
        """Compute combined decoder output vectors for a batch.

        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param qw_idxs (Tensor): padded question indices (b, q_len), where
                                       b = batch size, q_len = maximum question length 

        @returns combined_decoder_outputs (Tensor): combined output tensor  (b, q_len, h), where
                                        b = batch_size, q_len = maximum target sentence length,  h = hidden size
        """        
        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = self.enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, 1, self.hidden_size, device=self.device)
        
        enc_hiddens_proj = self.att_projection(self.enc_hiddens)

        dec_state = dec_init_state        #(batch_size, hidden_size)
        
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        # Initialize a list we will use to collect the combined decoder output o_t on each step
        combined_decoder_outputs = []

        for q_t in torch.split(q_emb, split_size_or_sections=1, dim=1):         #(batch_size, 1, hidden_size)
            Ybar_t = torch.cat((q_t, o_prev), dim=-1)                           #(batch_size, 1, 2*hidden_size)
            dec_state, o_t, _ = self.step(
                Ybar_t,
                dec_state,
                enc_hiddens_proj)
            combined_decoder_outputs.append(o_t)
    
        combined_decoder_outputs = torch.stack(combined_decoder_outputs, dim=1)       #(batch_size, q_len, hidden_size)
        logits = self.target_vocab_projection(combined_decoder_outputs)               #(batch_size, q_len, output_size)
        log_probs = F.log_softmax(logits, dim=-1)                                     #(batch_size, q_len, output_size)
        return dec_state, log_probs

    
    def step(self, 
            Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens_proj: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, 2*h). The input for the decoder,
                                where b = batch size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, c_len, h),
                                    where b = batch size, c_len = maximum context length, h = hidden size.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Not used outside of this function, simply returned for sanity check.
        """
        combined_output = None
        o_t, dec_state = self.decoder(Ybar_t, dec_state)                                            # o_t => (batch_size, 1, hidden_size)
        e_t = torch.bmm(o_t, torch.transpose(enc_hiddens_proj, dim0=1, dim1=2)).squeeze(dim=1)      # e_t => (batch_size, c_len)

        # Set e_t to -inf where enc_masks has 0
        if self.enc_masks is not None:
            e_t.data.masked_fill_(self.enc_masks == 0, -float('inf'))

        alpha_t = F.softmax(e_t, dim=-1)                                                            # alpha_t => (batch_size, c_len)
        a_t = torch.bmm(torch.unsqueeze(alpha_t, dim=1), self.enc_hiddens).squeeze(dim=1)           # a_t => (batch_size, 2*hidden_size)
        o_t = o_t.squeeze(1)                                                                        # o_t => (batch_size, hidden_size)
        U_t = torch.cat((o_t, a_t), dim=-1)                                                         # U_t => (batch_size, 3*hidden_size)
        V_t = self.combined_output_projection(U_t)                                                  # V_t => (batch_size, hidden_size) 
        O_t = self.dropout(torch.tanh(V_t))                                                         # O_t => (batch_size, hidden_size) 

        combined_output = O_t
        return dec_state, combined_output, e_t
