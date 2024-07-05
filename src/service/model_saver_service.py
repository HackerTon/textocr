import torch
from pathlib import Path
from os import remove


class ModelSaverService:
    def __init__(self, path: Path, topk: int = 2, name="default") -> None:
        self.model_directory = path
        self.topk = topk
        self.name = name
        self.latest_model = []

        if not self.model_directory.exists():
            self.model_directory.mkdir(parents=True)

    def _generate_save_name(self, epoch: int):
        return f"{epoch}_{self.name}_model.pt"

    def _checkAndExisting(self) -> bool:
        if len(self.latest_model) > self.topk:
            first_epoch_to_delete = self.latest_model.pop(0)
            model_to_delete = self.model_directory.joinpath(
                self._generate_save_name(first_epoch_to_delete)
            )
            remove(model_to_delete)
            print(f"{self._generate_save_name(first_epoch_to_delete)} removed!")

    def save_without_shape(self, model: torch.nn.Module, epoch: int):
        self._checkAndExisting()
        torch.save(
            model.state_dict(),
            self.model_directory.joinpath(self._generate_save_name(epoch)),
        )
        self.latest_model.append(epoch)

    def save_with_shape(self, model: torch.nn.Module, epoch: int):
        self._checkAndExisting()
        torch.save(
            model,
            self.model_directory.joinpath(self._generate_save_name(epoch)),
        )
        self.latest_model.append(epoch)
