import pytorch_lightning as pl
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

from streaming.vision import StreamingCIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    '''organize the data pipeline from accessing the data to loading it using PyTorch dataloaders'''


    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.local = '/tmp/karan_cifar10'
        self.remote = 's3://mosaicml-internal-dataset-cifar10/mds/2/'

    def prepare_data(self):
        # download the CIFAR-10 dataset
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    # PyTorch dataset instances
    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.cifar_train = StreamingCIFAR10(local=self.local, remote=self.remote,
                                                split='train', transform=self.transform,
                                                batch_size=self.batch_size,
                                                )
            # self.cifar_val = StreamingCIFAR10(local=self.local, remote=self.remote,
            #                                    split='val', transform=self.transform,
            #                                    batch_size=self.batch_size,
            #                                    prefix_int=2223)

    # dataloaders
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    # def val_dataloader(self):
    #    return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=1)

class CIFARLitModel(pl.LightningModule):
    '''model architecture, training, testing and validation loops'''
    def __init__(self, input_shape, num_classes, learning_rate=3e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # model architecture
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        n_sizes = self._get_output_shape(input_shape)

        # linear layers for classifier head
        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def _get_output_shape(self, shape):
        '''returns the size of the output tensor from the conv layers'''

        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._feature_extractor(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # computations
    def _feature_extractor(self, x):
        '''extract features from the conv blocks'''
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    def forward(self, x):
       '''produce final model output'''
       x = self._feature_extractor(x)
       x = x.view(x.size(0), -1)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = F.log_softmax(self.fc3(x), dim=1)
       return x

    # train loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # metric
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    # optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    # instantiate classes
    args = ArgumentParser()
    args.add_argument(
        '--n_devices',
        type=int,
        default=1,
        help='Number of gpus to use',
    )
    args.add_argument(
        '--n_nodes',
        type=int,
        default=1,
        help='Number of nodes to use',
    )
    args.add_argument(
        '--accelerator',
        type=str,
        default='gpu',
        help='Accelerator to use',
    )
    args.add_argument(
        '--strategy',
        type=str,
        default='ddp',
        help='Distributed strategy to use',
    )
    args = args.parse_args()
    np.random.seed(12341)
    n_devices = int(args.n_devices)

    import os
    os.environ['WORLD_SIZE'] = str(n_devices)
    os.environ['LOCAL_WORLD_SIZE'] = str(n_devices)
    os.environ['RANK'] = os.environ.get('LOCAL_RANK', '0')
    #print(f'Rank: {os.environ["RANK"]}')

    dm = CIFAR10DataModule(batch_size=32)
    dm.prepare_data()
    dm.setup()
    model = CIFARLitModel((3, 32, 32), dm.num_classes)
    # wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')


    # Initialize Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint()
    trainer = pl.Trainer(max_epochs=1,
                        devices=n_devices,
                        accelerator=args.accelerator,
                        strategy=args.strategy,
                        callbacks=[checkpoint_callback],
                        check_val_every_n_epoch=10,
                        num_nodes=int(args.n_nodes),
                        )

    # Train the model
    trainer.fit(model, dm)

    # wandb.finish()


