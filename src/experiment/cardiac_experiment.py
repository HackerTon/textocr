from typing import Tuple

import torch
from src.dataloader.dataset.container_dataset import CardiacDataset
from src.experiment.experiment_base import ExperimentBase
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import crop

from src.dataloader.dataloader import CardiacDatasetHDF5
from src.dataloader.transform import ToNormalized
from src.model.model import (
    BackboneType,
    MultiNet,
    MultiNetV2,
    UNETNetwork,
    FPNNetwork,
    MultiNetWithAttention,
)
from src.service.hyperparamater import Hyperparameter


class CardiacExperiment(ExperimentBase):
    def __init__(
        self, hyperparameter: Hyperparameter, device: str, model="multinet"
    ) -> None:
        super().__init__()

        self.train_dataloader, self.test_dataloader = (
            create_cardiac_dataloader_traintest(
                path=hyperparameter.data_path,
                batch_size=hyperparameter.batch_size_train,
            )
        )

        if model == "unet":
            self.model = UNETNetwork(numberClass=3)
        elif model == "multinet":
            self.model = MultiNet(numberClass=3, backboneType=BackboneType.RESNET50)
        elif model == "fpn":
            self.model = FPNNetwork(numberClass=3)
        elif model == "multinetv2":
            self.model = MultiNetV2(numberClass=3, backboneType=BackboneType.RESNET50)
        elif model == "multinetwithattention":
            self.model = MultiNetWithAttention(
                numberClass=3, backboneType=BackboneType.RESNET50
            )
        else:
            raise Exception(f"missing model {model}")

        self.preprocessor = v2.Compose(
            [
                ToNormalized(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Move weights to specified device
        self.model = self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=hyperparameter.learning_rate,
            fused=True if device == "cuda" else False,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=hyperparameter.learning_rate,
            steps_per_epoch=len(self.train_dataloader),
            epochs=hyperparameter.epoch,
        )


random_generator = torch.Generator().manual_seed(1234)


def trainc_collate_fn_with_random_size(data):
    if torch.rand(1, generator=random_generator)[0] > 0.5:
        current_size = 512
    else:
        current_size = 256
    images = []
    labels = []

    # If current_size is the same size as input
    # skip cropping
    if data[0][0].size(1) == current_size:
        for x in data:
            image, label = x
            images.append(image)
            labels.append(label)
    else:
        for x in data:
            image, label = x
            i, j, h, w = v2.RandomCrop.get_params(image, (current_size, current_size))
            images.append(crop(image, i, j, h, w))
            labels.append(crop(label, i, j, h, w))
    return (torch.stack(images), torch.stack(labels))


def train_collate(data):
    images = []
    labels = []
    for x in data:
        image, label = x
        images.append(image)
        labels.append(label)
    return (torch.stack(images), torch.stack(labels))


def create_cardiac_dataloader_traintest(
    path: str,
    batch_size: int,
    seed: int = 12345678,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    global_dataset = CardiacDataset(directory_path=path)
    SPLIT_PERCENTAGE = 0.8

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        global_dataset,
        [SPLIT_PERCENTAGE, 1 - SPLIT_PERCENTAGE],
        generator,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_dataloader, test_dataloader
