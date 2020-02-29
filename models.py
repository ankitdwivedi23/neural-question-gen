"""Top-level model classes.
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
import util

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
        device (string): cuda or cpu
    """
    def __init__(self, word_vectors, hidden_size, output_size, drop_prob=0., device='cpu'):
        super(Seq2Seq, self).__init__()
        
        self.device = device
        
        self.embedding = layers.Embedding(word_vectors=word_vectors,
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
        qw_idxs = qw_idxs[:-1]

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc, dec_init_state = self.encoder(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        
        decoder_outputs = []
        decoder_hidden = dec_init_state         #(batch_size, hidden_size)  
        for q_t in torch.split(q_emb, split_size_or_sections=1, dim=1):         #(batch_size, 1, hidden_size)    
            o_t, decoder_hidden = self.decoder(q_t, decoder_hidden)
            decoder_outputs.append(o_t)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)       #(batch_size, q_len, hidden_size)
        logits = self.projection(decoder_outputs)                   #(batch_size, q_len, output_size)
        log_probs = util.masked_softmax(logits, q_mask, dim=-1, log_softmax=True)       #(batch_size, q_len, output_size)
        
        return log_probs

    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                        We are simply returning this value so that we can sanity check
                                        your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t (be careful about the input/ output shapes!)
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        (dec_hidden, dec_cell) = self.decoder(Ybar_t, dec_state)
        dec_state = (dec_hidden, dec_cell)
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)), dim=2)
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh

        alpha_t = nn.functional.softmax(e_t, dim=1)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens), dim=1)
        U_t = torch.cat((dec_hidden, a_t), dim=1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t