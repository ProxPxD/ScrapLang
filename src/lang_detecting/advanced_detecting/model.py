import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import repeat_last, windowed
from more_itertools.more import padded
from pydash import chain as c, flow
from torch import Tensor

from src.lang_detecting.advanced_detecting.conf import Conf, ExpertConf
from src.lang_detecting.advanced_detecting.model_io_mging import Class, KindToTargets, KindToVocab, Vocab


class Expert(nn.Module):
    def __init__(self,
            vocab: Vocab,
            targets: list[Class],
            all_classes: list[Class],
            conf: ExpertConf,
            n_specs: int = 0,
        ):
        super().__init__()
        n_tokens, n_labels, n_layers = len(vocab), len(all_classes), len(conf.paddings)
        self.s_chunk = conf.chunking.size
        self.stride = conf.chunking.stride
        self.n_specs = n_specs

        self.embed = nn.Embedding(n_tokens, conf.emb_dim-n_specs, padding_idx=conf.padding_idx)
        channels = (conf.emb_dim, *conf.hidden_channels, n_labels)
        kernels = list(padded(conf.kernels, 3, len(channels)-2)) + [1]
        paddings = list(padded(conf.paddings, 0, len(channels)-1))
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=ci,
                out_channels=co,
                kernel_size=k,
                padding=p,
            ) for (ci, co), k, p in zip(windowed(channels, 2), kernels, paddings)
        ])
        self.hid_act = nn.LeakyReLU(negative_slope=conf.leaky_relu_slop)
        self.attn = nn.MultiheadAttention(embed_dim=channels[0], num_heads=1, batch_first=True)
        self.out_act = nn.Sigmoid()
        l_last_layer: int = self.s_chunk + sum(2*p - k + 1 for k, p in zip(kernels, paddings))
        self.positional = nn.Parameter(torch.tensor([0.4, 0.2, 0.4]))  # Beg, Mid, End
        output_mask = c(all_classes).map(targets.__contains__).map(int).value()
        self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.float32))

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
        assert 0 <= words.min() and words.max() < self.embed.num_embeddings, (words.min().item(), words.max().item(), self.embed.num_embeddings)
        x = self.embed(words)  # B x L x e0
        x = self._pad(x, -2, 0)
        specs = self._pad(specs, -2, 0)
        x = torch.cat([x, specs[..., :self.n_specs]], dim=-1)  # B x L x e1
        x = self.chunk_pad(x)  # B x ch x e1 x l_0
        B, ch, C, L = x.shape
        x = x.reshape(B*ch, C, L)
        for conv in self.convs:
            x = self.hid_act(conv(x))  # B*ch x c_k x l_k
        # x.permute()
        # x = self.attn(x)  # B*ch x o x l_k
        x = x.reshape(B, ch, *x.shape[-2:])   # B x ch x o x l_k
        #x = x.sum(dim=(-3, -1))
        x = self._weight_positional(x)  # B x o
        # Mask non-expert outputs
        x = x * self.output_mask + (self.output_mask - 1) * 1e9  # set masked to very negative value
        return x

    def _pad(self, tensor: Tensor, dim: int = -1, pad_val: int = 0) -> Tensor:
        s_unpadded = tensor.shape[dim]
        n_chunks = math.ceil(s_unpadded / self.s_chunk)
        overlap = self.s_chunk - self.stride
        overlap_offset = overlap if n_chunks > 1 else 0
        n_to_pad = (self.s_chunk * n_chunks - overlap_offset - s_unpadded)
        from_end = -(dim if dim < 0 else dim - len(tensor.shape))
        initial_zeroes = (0, 0) * (from_end - 1)
        tensor = F.pad(tensor, (*initial_zeroes, 0, n_to_pad), value=pad_val)
        return tensor

    def chunk_pad(self, x: Tensor) -> Tensor:
        x = x.contiguous()
        x = x.unfold(dimension=1, size=self.s_chunk, step=self.stride)
        return x

    def old_chuno(self, x):
        length_dim = -2
        n = x.size(length_dim)
        s_chunk, step = self.s_chunk, self.s_chunk_step
        shift_space = n - s_chunk
        n_fitting = shift_space // step + 1
        x_front = x.unfold(dimension=length_dim, size=s_chunk, step=step)  # B x ch x e x (l_0')
        front_end_idx = s_chunk + step * (n_fitting - 1) - 1
        if front_end_idx < n - 1:
            x_end = x[..., -s_chunk:, :].unsqueeze(-3).transpose(-2, -1)  # B x 1 x e x (l_0')
            x_out = torch.cat([x_front, x_end], dim=length_dim-1)  # B x 1 x e x l_0
        else:
            x_out = x_front
        return x_out  # B x ch x e x l_0

    def _weight_positional(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1, 3)  # B x c_k x ch x l_k
        x = x.flatten(-2)  # B x c_k x (ch*l_k)
        chunk_o_length = x.shape[-1]
        n_edge_vals = min(self.s_chunk_step // 2, chunk_o_length // 2)
        n_mid_vals = chunk_o_length - 2*n_edge_vals
        pos = self.positional / self.positional.sum()
        weights = torch.cat([pos[i].expand(n) for i, n in enumerate((n_edge_vals, n_mid_vals, n_edge_vals))]).to(dtype=x.dtype)
        weights = weights / weights.sum()
        x = (x * weights).sum(dim=-1)  # B x o
        return x


class Moe(nn.Module):
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            kinds_to_targets: KindToTargets,
            kind_to_specs: dict[str, Sequence[Callable]],
            conf: Conf,
        ):
        assert kinds_to_vocabs.keys() == kinds_to_targets.keys()
        super().__init__()
        all_targets = c(kinds_to_targets.values()).flatten().apply(flow(set, sorted)).value()
        self.n_classes = len(all_targets)
        self.experts = nn.ModuleList([
            Expert(vocabs, targets, all_targets, conf=conf.expert, n_specs=kind_to_specs.get(kind, 0))
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
