from collections import Counter
from dataclasses import dataclass, field
from typing import OrderedDict, Sequence, Optional

from sympy.core.cache import cached_property


@dataclass(frozen=True)
class Augment:
    is_augmenting: bool = True
    pre_augment_size: int = 5
    post_augment_size: int = pre_augment_size

@dataclass(frozen=True)
class Chunking:
    size: int = 2 ** 5
    overlap: int = 3

    @property
    def stride(self) -> int:
        return self.size - self.overlap

@dataclass(frozen=True)
class ValSet:
    size: float = .1
    min_n_label: int = 5

@dataclass(frozen=True)
class WordConstrain:
    len_thresh: int = 5
    n_non_uniq: int = 7
    len_to_uniq_ratio: float = 2.0

@dataclass
class Labels:
    all_names: tuple[str] = None  # Autofilled
    used_count: Counter[str] = None  # Autofilled

    @property
    def n_all(self) -> Optional[int]:
        return None if self.all_names is None else len(self.all_names)

    @property
    def n_used(self) -> Optional[int]:
        return None if self.used_count is None else len(self.used_names)

    @property
    def used_names(self) -> tuple[str, ...]:
        return tuple(self.used_count.keys())

@dataclass
class Data:
    min_record_n_thresh: int = 2**5
    chunking: Chunking = field(default_factory=Chunking)
    valset: ValSet = field(default_factory=ValSet)
    augment: Augment = field(default_factory=Augment)
    word: WordConstrain = field(default_factory=WordConstrain)
    labels: Labels = field(default_factory=Labels)


@dataclass
class ExpertConf:
    chunking: Chunking = field(default_factory=Chunking)
    padding_idx: int = None  # Autofill
    emb_dim: int = 32
    kernels: Sequence[int] = (3, 3, 3)
    hidden_channels: Sequence[int] = (64, 64, 64)
    paddings: Sequence[int] =  (0, 0, 0)
    leaky_relu_slop: float = 0.1

@dataclass(frozen=True)
class Weights:
    prob_tau: float = .5
    entropy_tau: float = .5

    freq_bias: float = 1.1 # 1.1 # 1.1
    neg_bias: float = 4 # 4  # 3

@dataclass(frozen=True)
class Train:
    epochs: int = 2**8  # 2**7
    lr: float = 1e-2  # 1e-5  # 1e-3
    weight_decay = 1e-5  # 1e-4
    max_batch_size: Optional[int] = 2**6
    accum_grad_bs: int = 2**9

@dataclass
class Supervision:
    cm_threshes: Sequence[float] = (.5, .66, .80, .90, .95, .99)
    metrics_thresh: float = .8

@dataclass
class Conf:
    seed: int = 9
    data: Data = field(default_factory=Data)
    expert: ExpertConf = field(default_factory=ExpertConf)
    train: Train = field(default_factory=Train)
    weights: Weights = field(default_factory=Weights)
    supervision: Supervision = field(default_factory=Supervision)
