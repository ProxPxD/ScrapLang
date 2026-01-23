import ast
import math
from collections import OrderedDict
from dataclasses import asdict
from typing import Sequence, Callable

import more_itertools
import toolz
import torch
from more_itertools import windowed
from pandas import DataFrame
import torch.nn as nn
from pydash import flow
from toolz import valmap
import pydash as _
from pydash import chain as c
from torch import Tensor, floor
import torch.nn.functional as F

from src.lang_detecting.advanced_detecting.conf import Conf, ExpertConf
from src.lang_detecting.advanced_detecting.model_io_mging import KindToOutputs, KindToSpecialGroup, Class, SpecialGroup, \
    ALL, TokenizedKindToGroupedVocab, KindToVocab, Vocab
from src.lang_detecting.preprocessing.data import LSC


class Expert(nn.Module):
    def __init__(self,
            vocab: Vocab,
            outputs: list[Class],
            all_classes: list[Class],
            conf: ExpertConf,
            n_specs: int = 0,
        ):
        super().__init__()
        n_tokens, n_classes, n_layers = len(vocab), len(all_classes), len(conf.paddings)
        self.register_buffer('s_chunk', torch.tensor(conf.s_chunk))
        self.register_buffer('s_chunk_step', torch.tensor(conf.s_chunk_step))
        output_mask = c(all_classes).map(outputs.__contains__).map(int).value()
        self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.int32))
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.norm = nn.Softmax()

        self.register_buffer('n_specs', torch.tensor(n_specs, dtype=torch.int32))
        self.embed = nn.Embedding(n_tokens, conf.emb_dim-n_specs)
        channels = (conf.emb_dim, *conf.hidden_channels, n_classes)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=ci,
                out_channels=co,
                kernel_size=k,
                padding=p,
            ) for (ci, co), k, p in zip(windowed(channels, 2), conf.kernels, conf.paddings)
        ])
        l_last_layer: int = conf.s_chunk + sum(2*p - k + 1 for k, p in zip(conf.kernels, conf.paddings))
        self.positional = nn.Parameter(Tensor([0.4, 0.2, 0.4]))  # Beg, Mid, End

    def forward(self, words: Tensor, specs: Tensor):
        """
        B - Batch size
        L - Length of the word with boundary padding
        e - Embedding dimension
        ch - Number of chunks
        l_k - Tensor length after k-th convolution [l_0 - Tensor length after chunking]
        c_k - k-th number of channels  [c_0 = e]
        """
        # words: B x L
        # specs: B x L x n_spec
        x = self.embed(words)  # B x L x e0
        x = torch.cat([x, specs[..., :self.n_specs]], dim=-1)  # B x L x e1
        x = self.chunk(x)  # B x ch x e1 x l_0
        B, ch, C, L = x.shape
        x = x.reshape(B*ch, C, L)
        for conv in self.convs:
            x = self.act(conv(x))  # B*ch x c_k x l_k

        x = x.reshape(B, ch, *x.shape[-2:])   # B x ch x c_k x l_k
        # merge '(ch)unks' and '(l_k)ength'
        x = self._weight_positional(x)  # B x o
        # Mask non-expert outputs
        x = self.norm(x * self.output_mask)
        return x

    def chunk(self, x: Tensor) -> Tensor:
        """
        #TODO
        """
        length_dim = -2
        n = x.size(length_dim)
        s_chunk, step = self.s_chunk.item(), self.s_chunk_step.item()
        shift_space = n - s_chunk
        if shift_space < 0:
            return self._pad_both_sides(x, -shift_space)
        n_fitting = shift_space // step + 1
        x_front = x.unfold(dimension=length_dim, size=s_chunk, step=step)  # B x ch x e x (l_0')
        front_end_idx = s_chunk + step * (n_fitting - 1) - 1
        if front_end_idx < n - 1:
            x_end = x[..., -s_chunk:, :].unsqueeze(-3).transpose(-2, -1)  # B x 1 x e x (l_0')
            x_out = torch.cat([x_front, x_end], dim=length_dim-1)  # B x 1 x e x l_0
        else:
            x_out = x_front
        return x_out  # B x ch x e x l_0

    @classmethod
    def _pad_both_sides(cls, x: Tensor, missing: int) -> Tensor:
        x_out = torch.stack([   # B x e x l_0 x ch(=2)
            F.pad(x, (missing, 0)),
            F.pad(x, (0, missing)),
        ])
        return x_out.transpose(-2, -1)  # B x ch(=2) x e x l_0

    def _weight_positional(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1, 3)  # B x c_k x ch x l_k
        x = x.flatten(-2)  # B x c_k x (ch*l_k)
        chunk_o_length = x.shape[-1]
        s_step = self.s_chunk_step.item()
        n_edge_vals = min(s_step, chunk_o_length // 2)
        n_mid_vals = chunk_o_length - 2*n_edge_vals
        pos = F.softplus(self.positional)
        weights = torch.cat([pos[i].expand(n)/n for i, n in enumerate((n_edge_vals, n_mid_vals, n_edge_vals))]).to(dtype=x.dtype)
        x = (x * weights).sum(dim=-1)  # B x o
        return x


class Moe(nn.Module):
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            kinds_to_outputs: KindToOutputs,
            kind_to_specs: dict[str, Sequence[Callable]],
            conf: Conf,
        ):
        assert  kinds_to_vocabs.keys() == kind_to_specs.keys()
        super().__init__()
        all_outputs = c(kinds_to_outputs.values()).flatten().apply(flow(set, sorted)).value()
        self.register_buffer('n_classes', torch.tensor(len(all_outputs)))
        self.experts = nn.ModuleList([
            Expert(vocabs, outputs, all_outputs, conf=conf.expert, n_specs=kind_to_specs.get(kind, 0))
            for (kind, vocabs), outputs in zip(kinds_to_vocabs.items(), kinds_to_outputs.values())
        ])

    def forward(self, kinds: Tensor, words: Tensor, specs: Tensor) -> Tensor:
        out = torch.zeros(words.size(0), self.n_classes.item(), device=words.device)
        for expert_idx, expert in enumerate(self.experts):
            mask = kinds == expert_idx
            if mask.any():
                # masked_words = words[mask].clone().contiguous()
                # masked_specs = specs[mask].clone().contiguous()
                # out[mask] = expert(masked_words, masked_specs)
                out[mask] = expert(words[mask], specs[mask])
        return out
