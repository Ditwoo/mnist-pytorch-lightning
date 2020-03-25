import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

from datasets import CSVDataset
from models import CNN


class MNISTModel(LightningModule):
    def __init__(self, hparams):
        super(MNISTModel, self).__init__()

        self.hparams = hparams
        self.model = CNN(n_classes=10)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        pred = self.forward(images)

        loss = self.ce_loss(pred, targets)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pred = self.forward(images)

        loss_val = self.ce_loss(pred, targets)

        labels = torch.argmax(pred, dim=1)
        val_acc = torch.sum(targets == labels).item() / (len(targets) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], []

    def train_dataloader(self):
        ds = CSVDataset(self.hparams.train_data)
        bs = self.hparams.batch_size
        nw = self.hparams.num_workers
        return DataLoader(ds, batch_size=bs, num_workers=nw)

    def val_dataloader(self):
        ds = CSVDataset(self.hparams.valid_data)
        bs = self.hparams.batch_size
        nw = self.hparams.num_workers
        return DataLoader(ds, batch_size=bs, num_workers=nw)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        parser.add_argument("--learning_rate", "--lr", default=0.001, type=float)

        # data
        parser.add_argument("--train_data", type=str, required=True)
        parser.add_argument("--valid_data", type=str, required=True)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        return parser


def main(hparams):
    model = MNISTModel(hparams)

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit
    )
    trainer.fit(model)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    parser = MNISTModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    main(hyperparams)
