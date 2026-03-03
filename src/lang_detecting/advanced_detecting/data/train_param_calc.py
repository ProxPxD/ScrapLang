from __future__ import annotations

import random
import time
from typing import Optional

import pydash as _
import torch
import torch.nn as nn
from torch import Tensor

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import TensorBatch


class TrainParamCalc:
    def __init__(self, conf: Conf, loss_func: nn.Module = None):
        self.conf: Conf = conf
        self.loss_func: Optional[nn.Module] = loss_func

    def compute_weights(self, bias: float = None) -> Tensor:
        bias = bias or self.conf.freq_bias
        all_labels = self.conf.all_label_names
        label_count = self.conf.used_label_count
        present_classes = label_count.keys()
        counts = torch.tensor([label_count[c] for c in present_classes])
        freq = counts / counts.sum()
        raw_weights = freq ** -bias
        raw_weights /= raw_weights.mean()

        present_idxs = torch.tensor(_.map_(present_classes, all_labels.index), dtype=torch.long)
        weights = torch.ones(len(all_labels), dtype=raw_weights.dtype)
        weights[present_idxs] = raw_weights
        return weights

    def shuffle_batches(self, batches: list[TensorBatch]) -> list[TensorBatch]:
        rng = random.Random()
        rng.seed(self.conf.seed or time.time())  # 9(3)
        rng.shuffle(batches)
        return batches

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = self.loss_func(logits, targets.float())
        final_loss = loss.mean()
        return final_loss
        # m_neg = targets == 0
        # loss_weights = torch.ones_like(preds)
        # loss_weights[m_neg] = (preds ** self.conf.neg_bias)[m_neg]
        # asym_loss = (loss * loss_weights)
        # loss_per_cls = loss.mean(0)
        # scale = loss_per_cls.detach().mean() / (loss_per_cls.detach() + eps)
        # m_pos = ~m_neg
        # final_loss_full = asym_loss * (1 + m_pos * (scale - 1))
        # final_loss = final_loss_full.mean()
        # final_loss.backward()
