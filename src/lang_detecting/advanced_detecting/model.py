import math
from collections.abc import Sequence
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import windowed
from more_itertools.more import padded
from pydash import chain as c
from pydash import flow
from torch import Tensor

from src.lang_detecting.advanced_detecting.conf import Conf, ExpertConf
from src.lang_detecting.advanced_detecting.model_io_mging import Class, KindToTargets, KindToVocab, Vocab


class Expert(nn.Module):
    def __init__(self,
            vocab: list[Vocab],
            targets: list[Class],
            all_classes: list[Class],
            conf: ExpertConf,
            n_specs: int = 0,
        ):
        super().__init__()
        n_tokens, n_labels = len(vocab), len(all_classes)
        self.funcs = {
            (LOGSUMEXP:='logsumexp'): torch.logsumexp,
            'relu': torch.relu,
            (SUM:='sum'): torch.sum,
        }
        self.s_chunk = conf.chunking.size
        self.stride = conf.chunking.stride
        self.n_specs = n_specs

        self.embed = nn.Embedding(n_tokens, conf.emb_dim - n_specs, padding_idx=conf.padding_idx)
        channels = (conf.emb_dim, *conf.hidden_channels)
        kernels = [*list(padded(conf.kernels, 3, len(channels) - 2)), 1]
        paddings = list(padded(conf.paddings, 0, len(channels) - 1))
        self.conv_dropout = nn.Dropout(p=conf.p_conv_dropout)
        self.dropout = nn.Dropout(p=conf.p_dropout)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=ci,
                out_channels=co,
                kernel_size=k,
                padding=p,
            ) for (ci, co), k, p in zip(windowed(channels, 2), kernels, paddings)
        ])
        self.hid_act = nn.Tanh()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels[-1],
            num_heads=1,
            batch_first=True,
            dropout=conf.p_dropout,
        )
        self.post_attn_pool_name = SUM
        self.norm = nn.LayerNorm(channels[-1], eps=1e-3)
        self.post_attn_classifier = nn.Linear(channels[-1], n_labels)
        output_mask = c(all_classes).map(targets.__contains__).map(int).value()
        self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.float32))

    @property
    def post_attn_pool(self) -> Callable:
        return self.funcs[self.post_attn_pool_name]

    def forward(self, words: Tensor, specs: Tensor) -> Tensor:
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
        if words.min() < 0 or words.max() >= self.embed.num_embeddings:
            v_min, v_max, n_embed = words.min().item(), words.max().item(), self.embed.num_embeddings
            raise ValueError(f'{v_min=}, {v_max=}, {n_embed=}')

        x = self.dropout(self.embed(words))  # B x L x e0
        x = self._pad(x, -2, 0)
        specs = self._pad(specs, -2, 0)
        x = torch.cat([x, specs[..., : self.n_specs]], dim=-1)  # B x L x e1
        x = self.chunk(x)  # B x ch x e1 x l_0
        B, ch, C, L = x.shape
        x = x.reshape(B * ch, C, L)
        for conv in self.convs:
            x = self.hid_act(conv(x))
            x = self.conv_dropout(x)  # B*ch x c_k x l_k
        x = x.permute(0, 2, 1)  # B*ch x l_k x c_k
        *_, L, C = x.shape
        #x_norm = x
        x_norm = self.norm(x)
        attn_mask, _ = self.attn(x_norm, x_norm, x_norm)  # B*ch x l_k x c_k
        attn_mask = self.norm(attn_mask)
        gate = torch.sigmoid(attn_mask)
        x = (x * gate).reshape(B, ch * L, C)  # B x ch*l_k x c_k
        x = self.post_attn_pool(x, dim=-2)  # B x c_k
        x = self.post_attn_classifier(x)  # B x o
        # Mask non-expert outputs
        x = x.masked_fill(self.output_mask == 0, -1e9)  # set masked to very negative value
        return x

    def _pad(self, tensor: Tensor, dim: int = -1, pad_val: int = 0) -> Tensor:
        s_unpadded = tensor.shape[dim]
        n_chunks = math.ceil(s_unpadded / self.s_chunk)
        overlap = self.s_chunk - self.stride
        overlap_offset = overlap if n_chunks > 1 else 0
        n_to_pad = self.s_chunk * n_chunks - overlap_offset - s_unpadded
        from_end = -(dim if dim < 0 else dim - len(tensor.shape))
        initial_zeroes = (0, 0) * (from_end - 1)
        tensor = F.pad(tensor, (*initial_zeroes, 0, n_to_pad), value=pad_val)
        return tensor

    def chunk(self, x: Tensor) -> Tensor:
        x = x.contiguous()
        x = x.unfold(dimension=1, size=self.s_chunk, step=self.stride)
        return x


class Moe(nn.Module):
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            kinds_to_targets: KindToTargets,
            kind_to_specs: dict[str, Sequence[Callable]],
            conf: Conf,
        ):
        if kinds_to_vocabs.keys() != kinds_to_targets.keys():
            raise ValueError(f'Incompatible kinds to vocab/targets mapping: {list(kinds_to_targets.keys())} {list(kinds_to_targets.keys())}')
        super().__init__()
        all_targets = c(kinds_to_targets.values()).flatten().apply(flow(set, sorted)).value()  # type: ignore[arg-type]
        self.n_classes = len(all_targets)
        self.experts: nn.ModuleList[Expert] = nn.ModuleList([
            Expert(vocabs, targets, all_targets, conf=conf.expert, n_specs=kind_to_specs.get(kind, 0),)
            for (kind, vocabs), targets in zip(kinds_to_vocabs.items(), kinds_to_targets.values())
        ])

    def forward(self, kinds: Tensor, words: Tensor, specs: Tensor) -> Tensor:
        out = torch.zeros(words.size(0), self.n_classes, device=words.device)
        specs = specs.to_dense()
        for expert_idx, expert in enumerate(self.experts):
            mask = kinds == expert_idx
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0]
                out[idx] = expert(
                    words.index_select(0, idx),
                    specs.index_select(0, idx),
                )
        return out
