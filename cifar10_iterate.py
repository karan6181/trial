import argparse
from typing import Callable, Any

from torch.utils.data import DataLoader
from torchvision import transforms

from streaming.base import StreamingDataset

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str)
    parser.add_argument("--remote_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--drop_last", type=bool, default=False)
    args = parser.parse_args()

    local_dir = args.local_dir
    remote_dir = args.remote_dir
    train_dataloader = get_dataloader(remote_dir, local_dir, args.shuffle, args.batch_size)

    for idx, batch in enumerate(train_dataloader):
        print(idx)

