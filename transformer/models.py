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
import copy

from typing import List, Tuple, Dict, Set, Union

class TransformerModel(nn.Module):
    """Transformer model based on the paper “Attention Is All You Need”.
    Paper -  https://arxiv.org/pdf/1706.03762.pdf
    Video - https://youtu.be/nfs8NYg7yQM
    Actual Video (No kidding) - https://youtu.be/5vcj8kSwBCY

    Layers:
        Embedding Layer
        Encoder - Stack of encoder layers
        Decoder - Stack of decoder layers
        Generator - Final (FC + softmax) layer
    Args:
        vocab_size (int): Vocab size (output dimension)
        device (str): 'cuda:0' or 'cpu'
        d_model (int): Number of expected features in the encoder/decoder inputs
        nhead (int):  Number of heads in the multiheadattention models
        num_encoder_layers (int): Number of logits for softmax layer
        num_decoder_layers (int): 'cuda:0' or cpu
        dim_feedforward (int): Dimension of the feedforward network model
        dropout (float): Dropout probability
    """
    def __init__(self, vocab_size, device, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        c = copy.deepcopy
        position = layers.PositionalEncoding(d_model, dropout)
        self.device = device
        self.src_embed = nn.Sequential(layers.TransformerEmbedding(d_model, vocab_size), c(position))
        self.tgt_embed = nn.Sequential(layers.TransformerEmbedding(d_model, vocab_size), c(position))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.generator = layers.Generator(d_model, vocab_size)
        self.model_type = 'transformer'

        self._reset_parameters()
    
    def forward(self, cw_idxs, qw_idxs, c_mask, q_mask):
        """
        Args:
            cw_idxs (tensor): Padded context sequence. Shape: (batch_size, c_len)
            qw_idxs (tensor): Padded question sequence. Shape (batch_size, q_len)
            c_mask (tensor):  Context mask. Shape: (batch_size, c_len)
            q_mask (tensor): Question mask. Shape: (batch_size, q_len)
        Returns:
            log_p (tensor): log of softmax distribution of linear projection of decoder output. Shape: 
        """
        enc_out = self.encode(cw_idxs, c_mask)                      # (batch_size, c_len, d_model)
        log_p = self.decode(qw_idxs, enc_out, c_mask, q_mask)       # (batch_size, q_len, vocab_size)
        return log_p    

    def encode(self, cw_idxs, c_mask):
        c_emb = self.src_embed(cw_idxs)                             # (batch_size, c_len, d_model)
        c_emb = c_emb.transpose(0,1)                                # (c_len, batch_size, d_model)

        enc_out = self.encoder(c_emb, src_key_padding_mask=c_mask)  # (c_len, batch_size, d_model)
        return enc_out
    
    def decode(self, qw_idxs, enc_out, c_mask, q_mask):
        q_emb = self.tgt_embed(qw_idxs)                             # (batch_size, q_len, d_model)
        q_emb = q_emb.transpose(0,1)                                # (q_len, batch_size, d_model)

        self_attn_mask = self.generate_square_subsequent_mask(qw_idxs.size(1)).to(device=self.device)    # (q_len, q_len)

        dec_out = self.decoder(
            tgt=q_emb,
            memory=enc_out,
            tgt_mask=self_attn_mask,
            tgt_key_padding_mask=q_mask,
            memory_key_padding_mask=c_mask)     # (q_len, batch_size, d_model)
        log_p = self.generator(dec_out)         # (q_len, batch_size, vocab_size)        
        log_p = log_p.transpose(0,1)
        return log_p                            # (batch_size, q_len, vocab_size)
    

    def _reset_parameters(self):
        "Initiate parameters in the transformer model."
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, size):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask

