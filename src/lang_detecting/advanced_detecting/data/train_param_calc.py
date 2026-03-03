from __future__ import annotations

import random
import time

import pydash as _
import torch
from torch import Tensor

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import TensorBatch


class TrainParamCalc:
    def __init__(self, conf: Conf):
        self.conf: Conf = conf

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
