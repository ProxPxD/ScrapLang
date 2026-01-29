from functools import cached_property
from itertools import product, repeat
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame
from pydash import chain as c, flow
from toolz import valmap
from torch import Tensor

from src.constants import Paths
from src.context import Context
from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.dataset import BucketChunkDataset
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import KindToTokenMgr, ModelIOMgr
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer
from src.resouce_managing.valid_data import ValidDataMgr

EXCEPTION = None
try:
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    from torchmetrics import ConfusionMatrix, Metric, Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
    import matplotlib.pyplot as plt
    HAS_TRAINING_SUPERVISION = True
except ImportError as e:
    EXCEPTION = e
    HAS_TRAINING_SUPERVISION = False

class AdvancedDetector:
    def __init__(self, context: Context, lang_script: DataFrame, valid_data_mgr: ValidDataMgr, conf: Conf):
        self.context =  context
        self.model_io_mgr = ModelIOMgr()
        self.valid_data_mgr = valid_data_mgr
        self.conf = conf

        kinds_to_tokens_classes = self.model_io_mgr.extract_kinds_to_vocab_classes(lang_script)
        self.model_io_mgr.update_model_io_if_needed(kinds_to_tokens_classes)
        kinds_to_vocab, kinds_to_targets = KindToTokenMgr.separate_kinds_tos(kinds_to_tokens_classes)
        targets = c(kinds_to_targets.values()).flatten().sorted_uniq().value()
        kind_to_specs: dict[str, Sequence[Callable]] = {
            'Latn': [str.isupper],
            'Cyrl': [str.isupper],
        }
        self.tokenizer = MultiKindTokenizer(kinds_to_vocab, targets, kind_to_specs=kind_to_specs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._class_names = c(kinds_to_targets.values()).flatten().apply(flow(set, sorted)).value()
        self.moe = Moe(kinds_to_vocab, kinds_to_targets, valmap(len, kind_to_specs), conf=self.conf).to(self.device)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.writer = None
        self.metrics = {}
        self.confusion_matrix = None
        self.init_for_training()

    def init_for_training(self) -> None:
        if self.context.dev and not HAS_TRAINING_SUPERVISION:
            raise RuntimeError('Dev mode run without tensorboard or torchmetrics or tqdm or matplotlib installed') from EXCEPTION
        if self.dev_training:
            Paths.DETECTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=Paths.DETECTION_LOG_DIR) if self.dev_training else None
        self.metrics: dict[str, Metric] = {
            f'{metric_class.__name__}_{mode}'.lower(): metric_class(task='multiclass', num_classes=self._n_classes, average=mode).to(self.device)
            for metric_class in [Accuracy, Precision, Recall, F1Score]
            for mode in ('macro', )
        }  if self.dev_training else {}
        self.metrics['matthews_corr_coef'] = MatthewsCorrCoef(task='multiclass', num_classes=self._n_classes).to(self.device) if self.dev_training else None
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self._n_classes).to(self.device) if self.dev_training else None

    @property
    def dev_training(self) -> bool:
        return self.context.dev and HAS_TRAINING_SUPERVISION

    @cached_property
    def _n_classes(self) -> int:
        return len(self._class_names)

    def retrain_model(self):
        self.init_for_training()
        dataset = BucketChunkDataset(self.valid_data_mgr.data, tokenizer=self.tokenizer, conf=self.conf, shuffle=True)
        optimizer = torch.optim.AdamW(self.moe.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        for epoch in range(self.conf.epochs):
            self._reset_metrics()
            total_loss = 0.0
            n_records = 0
            for batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{self.conf.epochs}"):
                kinds, words, specs, targets = [t.to(self.device) for t in batch]
                n_records += (bs:=words.size(0))
                preds: Tensor = self.moe(kinds, words, specs)
                loss = self.loss_func(preds, targets)
                self._update_metrics(preds, targets)
                loss.backward()
                total_loss += loss.item()

                if n_records >= self.conf.accum_grad_bs:
                    optimizer.step()
                    optimizer.zero_grad()
                    n_records = 0
            if n_records:
                optimizer.step()
            self._board_metrics(epoch)

    def _manage_metrics(self, func_name: str, *args, **kwargs) -> None:
        for metric in self.metrics.values():
            getattr(metric, func_name)(*args, **kwargs)

    def _reset_metrics(self) -> None:
        if not self.dev_training:
            return
        self._manage_metrics('reset')
        self.confusion_matrix.reset()

    def _update_metrics(self, preds: Tensor, targets: Tensor) -> None:
        if not self.dev_training:
            return
        self._manage_metrics('update', preds, targets)
        pred_labels, true_labels = preds.argmax(dim=1), targets.argmax(dim=1)
        self.confusion_matrix.update(pred_labels, true_labels)

    def _board_metrics(self, epoch: int) -> None:
        if not self.dev_training:
            return
        for name, metric in self.metrics.items():
            metric: Metric
            self.writer.add_scalar(f'train/metric/{name}', metric.compute().item(), epoch)
        self._board_confusion_matrix(epoch, 'train')

    def _board_confusion_matrix(self, epoch: int, mode: str):
        if not self.dev_training:
            return
        confusion_matrix = self.confusion_matrix.compute().cpu().numpy()
        fig = self.plot_confusion_matrix(confusion_matrix, self._class_names)
        self.writer.add_figure(f'ConfusionMatrix/{mode}', fig, epoch)
        plt.close(fig)

    @classmethod
    def plot_confusion_matrix(cls, conf_mat, class_names):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(conf_mat, interpolation="nearest")

        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
            xticklabels=class_names, yticklabels=class_names,
            ylabel='True label', xlabel='Predicted label',
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        thresh = conf_mat.max() / 2
        n_classes = conf_mat.shape[0]
        for i, j in product(range(n_classes), repeat=2):
            ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='white' if conf_mat[i, j] > thresh else 'black')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        fig.tight_layout()
        return fig
