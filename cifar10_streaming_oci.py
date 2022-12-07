import os
import argparse
import random
from typing import Callable, Any, Tuple

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import Backend
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from streaming.base import StreamingDataset

# python -m torch.distributed.launch --nproc_per_node=1 cifar10_streaming_oci.py

class CIFAR10Dataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable
                ) -> None:
        super().__init__(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        x = obj['x']
        y = obj['y']
        return self.transforms(x), y

def get_dataloader(remote_train, local_train, shuffle_train, batch_size):
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    ])

    train_dataset = CIFAR10Dataset(remote_train, local_train, shuffle_train, batch_size=batch_size, transforms=transformation)
    # test_dataset  = CIFAR10Dataset(remote_test, local_test, shuffle_test, batch_size=batch_size, transforms=transformation)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,
                            pin_memory=True,
                            persistent_workers=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def fit(model: nn.Module, train_dataloader: DataLoader, device) -> Tuple[float, float]:
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for imgs, labels in tepoch:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            labels_hat = model(imgs)
            loss = criterion(labels_hat, labels)
            train_running_loss += loss.item()
            _, preds = torch.max(labels_hat.data, 1)
            train_running_correct += (preds == labels).sum().item()
            loss.backward()
            optimizer.step()

    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)

    return train_loss, train_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_dir", type=str)
    parser.add_argument("--remote_dir", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--drop_last", type=bool, default=False)
    args = parser.parse_args()

    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'


    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if you are suing GPU
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    rank = args.local_rank
    world_size = torch.cuda.device_count()

    torch.cuda.set_device(rank)
    dist_url = "env://"
    torch.distributed.init_process_group(backend=Backend.NCCL, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    # device = torch.device(f'cuda:{rank}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_dir = args.local_dir
    remote_dir = args.remote_dir
    train_dataloader = get_dataloader(remote_dir, local_dir, args.shuffle, args.batch_size)

    model = Net().to(rank)
    model = DDP(model, device_ids=[rank])
    #model = model.to(device)
    #model = nn.DataParallel(model, device_ids=[0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epochs):
        train_epoch_loss, train_epoch_accuracy = fit(model, train_dataloader, device)
        print(f'epoch: {epoch+1}/{args.epochs} Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}')
    torch.distributed.destroy_process_group()
