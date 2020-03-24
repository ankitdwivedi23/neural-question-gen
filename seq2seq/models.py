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
import random

from typing import List, Tuple, Dict, Set, Union

PAD = 0
SOS = 2
EOS = 3


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
        output_size(int): Number of logits for softmax layer
        device (string): 'cuda:0' or 'cpu'
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, output_size, device, drop_prob=0., num_layers=1):
        super(Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.word_vectors = word_vectors
        self.model_type = 'seq2seq'

        #self.emb = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        #self.dropout = nn.Dropout(p=drop_prob)
        self.emb = layers.Embedding(word_vectors, hidden_size, drop_prob=drop_prob)
        
        self.encoder = layers.EncoderRNN(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     drop_prob=drop_prob)

        self.decoder = layers.DecoderRNN(input_size=hidden_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        drop_prob=drop_prob)      

        self.generator = layers.Generator(hidden_size, output_size)


    def forward(self, cw_idxs, qw_idxs):
        _, dec_init_state = self.encode(cw_idxs)
        _, log_probs = self.decode(dec_init_state, qw_idxs)         #(batch_size, q_len, output_size)
        return log_probs


    def encode(self, cw_idxs):
        c_mask = cw_idxs != PAD
        c_len = c_mask.sum(-1)

        c_emb = self.emb(cw_idxs)                                   # (batch_size, c_len, hidden_size)

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
        #q_emb = self.dropout(q_emb)

        # Initialize a list we will use to collect the combined decoder output o_t on each step
        combined_decoder_outputs = []

        for q_t in torch.split(q_emb, split_size_or_sections=1, dim=1):         #(batch_size, 1, hidden_size)
            o_t, dec_state = self.decoder(q_t, dec_state)                       #(batch_size, 1, hidden_size)
            o_t = o_t.squeeze(1)                                                #(batch_size, hidden_size)
            combined_decoder_outputs.append(o_t)
    
        combined_decoder_outputs = torch.stack(combined_decoder_outputs, dim=1)       #(batch_size, q_len, hidden_size)
        log_probs = self.generator(combined_decoder_outputs)       #(batch_size, q_len, output_size)
        return dec_state, log_probs        

##################################################################################################################

class Seq2SeqAttn(nn.Module):
    """Seq2seq model with attention
    Implements a basic seq2seq network (with attention):
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Decoder layer: Decode the encoded sequence word by word.
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
        self.word_vectors = word_vectors
        self.device = device
        self.enc_hiddens = None
        self.enc_masks = None
        self.model_type = 'seq2seq_attn'
        
        #self.emb = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        self.emb = layers.Embedding(word_vectors, hidden_size)

        self.encoder = layers.EncoderRNN(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     drop_prob=drop_prob)

        self.decoder = layers.DecoderRNN(input_size=2*hidden_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        drop_prob=drop_prob)

        self.att_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(in_features=3*hidden_size, out_features=hidden_size, bias=False)
        self.generator = layers.Generator(hidden_size, output_size)
        self.dropout = nn.Dropout(p=drop_prob)
    
    
    def forward(self, cw_idxs, qw_idxs):
        _, dec_init_state = self.encode(cw_idxs)
        _, log_probs = self.decode(dec_init_state, qw_idxs)         # (batch_size, q_len, output_size)
        return log_probs

    
    def encode(self, cw_idxs):
        c_mask = cw_idxs != PAD
        c_len = c_mask.sum(-1)

        c_emb = self.emb(cw_idxs)                                   # (batch_size, c_len, hidden_size)

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
        #q_emb = self.dropout(q_emb)

        # Initialize a list we will use to collect the combined decoder output o_t on each step
        combined_decoder_outputs = []

        for q_t in torch.split(q_emb, split_size_or_sections=1, dim=1):         #(batch_size, 1, hidden_size)
            Ybar_t = torch.cat((q_t, o_prev), dim=-1)                           #(batch_size, 1, 2*hidden_size)
            dec_state, o_t, _ = self.step(
                Ybar_t,
                dec_state,
                self.enc_hiddens,
                enc_hiddens_proj,
                self.enc_masks)
            combined_decoder_outputs.append(o_t.squeeze(dim=1))
            o_prev = o_t

        combined_decoder_outputs = torch.stack(combined_decoder_outputs, dim=1)       #(batch_size, q_len, hidden_size)
        log_probs = self.generator(combined_decoder_outputs)                          #(batch_size, q_len, output_size)
        return dec_state, log_probs

    
    def step(self, 
            Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks=None) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
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
        if enc_masks is not None:
            e_t = e_t.masked_fill(self.enc_masks == 0, float('-inf'))
            #e_t.data.masked_fill_(self.enc_masks == 0, -float('inf'))

        alpha_t = F.softmax(e_t, dim=-1)                                                            # alpha_t => (batch_size, c_len)
        
        a_t = torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens)                               # a_t => (batch_size, 1, 2*hidden_size)
        
        U_t = torch.cat((o_t, a_t), dim=-1)                                                         # U_t => (batch_size, 1, 3*hidden_size)
        V_t = self.combined_output_projection(U_t)                                                  # V_t => (batch_size, 1, hidden_size) 
        O_t = self.dropout(torch.tanh(V_t))                                                         # O_t => (batch_size, 1, hidden_size) 

        combined_output = O_t
        return dec_state, combined_output, e_t

##################################################################################################################

class Seq2SeqGru(nn.Module):
    """Baseline seq2seq model
    Implements a basic seq2seq network (using GRU):
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
    def __init__(self, Idx2Word, hidden_size, output_size, device, drop_prob=0.2, learning_rate=0.01, num_layers=1):
        super(Seq2SeqGru, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.Idx2Word = Idx2Word
        self.model_type = 'seq2seqGru'
        self.teacher_forcing_ratio = 0.5
        self.SOS_token = 2
        self.EOS_token = 3
        self.EOS = "--EOS--"

        self. criterion = nn.NLLLoss(reduction='sum')
        self.encoder = layers.EncoderRNNCell(input_size=hidden_size, output_size=output_size,
                                     hidden_size=hidden_size,
                                     device=device)

        self.decoder = layers.DecoderSimpleRNN(input_size=hidden_size, output_size=output_size,
                                        hidden_size=hidden_size,
                                        device=device)       
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.generator = layers.Generator(hidden_size, output_size)

    def forward(self, cw_idxs, qw_idxs):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_tensor, target_tensor = cw_idxs, qw_idxs

        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1)
        batch_size = cw_idxs.size(0)

        loss = 0
        encoder_hidden = self.encoder.initHidden(batch_size), self.encoder.initHidden(batch_size)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[:, ei].unsqueeze(0), encoder_hidden)
            encoder_hidden = encoder_hidden[0].unsqueeze(0), encoder_hidden

        decoder_input = torch.tensor([[self.SOS_token]*batch_size], device=self.device)
        
        decoder_hidden =  encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                predicted = self.generator(decoder_output)
                loss += self.criterion(predicted.squeeze(0), target_tensor[:, di])
                decoder_input = target_tensor[:, di].unsqueeze(0)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(2).detach()  # detach from history as input
                predicted = self.generator(decoder_output)
                loss += self.criterion(predicted.squeeze(0), target_tensor[:, di])
                #if decoder_input.item() == self.EOS_token:
                #    break

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return loss.item() / target_length
    
    def evaluate(self, cw_idxs, max_length=2):
        with torch.no_grad():
            input_tensor = cw_idxs
            input_length = input_tensor.size(1)
            batch_size = cw_idxs.size(0)

            encoder_hidden = self.encoder.initHidden(batch_size), self.encoder.initHidden(batch_size)
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[:, ei].unsqueeze(0), encoder_hidden)
                encoder_hidden = encoder_hidden[0].unsqueeze(0), encoder_hidden

            decoder_input = torch.tensor([[self.SOS_token]*batch_size], device=self.device)
            decoder_hidden =  encoder_hidden

            decoded_words = []

            for di in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.EOS_token:
                    decoded_words.append(self.EOS)
                    break
                else:
                    decoded_words.append(self.Idx2Word[topi.item()])

                decoder_input = topi.squeeze(2).detach()

            return decoded_words[0:1]

