"""Inspired and adapted from FastMETRO (https://github.com/postech-ami/FastMETRO/blob/main/src/modeling/model/transformer.py) 
and T2MGPT (https://github.com/Mael-zys/T2M-GPT/blob/main/models/t2m_trans.py)"""

import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor
import math


class Transformer(nn.Module):
    """Transformer encoder-decoder"""

    def __init__(
        self,
        model_dim=512,
        nhead=8,
        num_enc_layers=3,
        num_dec_layers=3,
        feedforward_dim=2048,
        dropout=0.1,
        activation="relu",
        dict_size=512,
        block_size=103,
        autoreg=True,
    ):
        """
        Parameters:
            - model_dim: The hidden dimension size in the transformer architecture
            - nhead: The number of attention heads in the attention modules
            - num_enc_layers: The number of encoder layers in the transformer encoder
            - num_dec_layers: The number of decoder layers in the transformer decoder
            - feedforward_dim: The hidden dimension size in MLP
            - dropout: The dropout rate in the transformer architecture
            - activation: The activation function used in MLP
            - dict_size: The number of vectors in the Mesh-VQ-VAE codebook (Only used if autoreg is True)
            - block_size: The total number of elements for the decoder (img + camrot + mesh) (Only used if autoreg is True)
        """
        super().__init__()
        self.model_dim = model_dim
        self.nhead = nhead

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            model_dim, nhead, feedforward_dim, dropout, activation
        )
        encoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)

        # transformer decoder
        decoder_layer = TransformerDecoderLayer(
            model_dim, nhead, feedforward_dim, dropout, activation
        )
        decoder_norm = nn.LayerNorm(model_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers, decoder_norm)

        self.autoreg = autoreg
        if autoreg:
            self.dec_head = CrossCondTransHead(
                dict_size,
                model_dim,
                block_size,
                num_dec_layers,
                nhead,
                dropout,
                feedforward_dim,
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_features, mesh_tokens, pos_embed):

        # Transformer Encoder
        enc_img_features = self.encoder(
            img_features, pos=pos_embed
        )  # (1 + height * width) X batch_size X feature_dim

        # Transformer Decoder
        zero_tgt = torch.zeros_like(mesh_tokens)  # num_mesh X batch_size X feature_dim
        mesh_features = self.decoder(
            mesh_tokens,
            enc_img_features,
            pos=pos_embed,
            query_pos=zero_tgt,
        )  # num_mesh X batch_size X feature_dim

        if not self.autoreg:
            return enc_img_features, mesh_features

        mesh_logits = self.dec_head(mesh_features)

        return enc_img_features, mesh_logits


class TransformerEncoder(nn.Module):
    """Transformer encoder"""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(
        self,
        src,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Transformer decoder"""

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                pos=pos,
                query_pos=query_pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""

    def __init__(
        self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # tensor[0] is for a camera token (no positional encoding)
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer"""

    def __init__(
        self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class CausalCrossConditionalSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        block_size=16,
        n_head=8,
        drop_out_rate=0.1,
        feedforward_dim=2048,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(
            embed_dim, block_size, n_head, drop_out_rate
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransHead(nn.Module):
    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        block_size=54,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        feedforward_dim=2048,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, block_size, n_head, drop_out_rate, feedforward_dim)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu, not {activation}.")


def build_transformer(transformer_config, autoreg=True):
    return Transformer(
        model_dim=transformer_config["model_dim"],
        dropout=transformer_config["dropout"],
        nhead=transformer_config["nhead"],
        feedforward_dim=transformer_config["feedforward_dim"],
        num_enc_layers=transformer_config["num_enc_layers"],
        num_dec_layers=transformer_config["num_dec_layers"],
        dict_size=transformer_config["dict_size"],
        block_size=transformer_config["block_size"],
        autoreg=autoreg,
    )
