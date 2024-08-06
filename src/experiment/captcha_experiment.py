from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split

from src.experiment.experiment_base import ExperimentBase
from src.model.model import TextOcrModel
from src.service.hyperparamater import Hyperparameter
from src.dataloader.dataset.captcha_dataset import CaptchaDataset


class TextocrExperiment(ExperimentBase):
    def __init__(self, hyperparameter: Hyperparameter, device: str) -> None:
        super().__init__()

        # Initialization
        self.train_dataloader, self.test_dataloader = self.create_dataloader(
            path=hyperparameter.data_path,
            batch_size=hyperparameter.batch_size_train,
        )
        self.model = TextOcrModel()

        # Move weights to specified device
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=hyperparameter.learning_rate,
            fused=True if device == "cuda" else False,
        )

    @staticmethod
    def collate_fn(data):
        batched_image = []
        batched_label = []
        for image, label in data:
            batched_image.append(image)
            batched_label.append(label)
        return batched_image, batched_label

    def create_dataloader(
        self,
        path: str,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader]:
        dataset = CaptchaDataset(path)
        length_dataset = len(dataset)
        train_length = int(length_dataset * 0.8)
        test_length = length_dataset - train_length
        train_dataset, test_dataset = random_split(
            dataset,
            [train_length, test_length],
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
        )
        return train_dataloader, test_dataloader


if __name__ == "__main__":
    hyperparameter = Hyperparameter(
        None, 0.05, 10, 10, "project-2-at-2024-08-06-07-03-5a7af7f3.json"
    )
    experiment = TextocrExperiment(hyperparameter, "cpu")
    for image, label in experiment.train_dataloader:
        break
