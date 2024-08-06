import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import v2

from src.experiment.captcha_experiment import CaptchaExperiment
from src.service.hyperparamater import Hyperparameter
from src.service.model_saver_service import ModelSaverService

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        name: str,
        hyperparameter: Hyperparameter,
        train_report_rate: float = 0.001,
    ):
        """
        train_report_rate: float = [0.0, 1.0]
        """

        torch.manual_seed(123456)

        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        directory_name = "data/model/{}_{}".format(
            timestamp,
            name.replace(" ", "_"),
        )
        self.writer_train = SummaryWriter(f"{directory_name}/train")
        self.writer_test = SummaryWriter(f"{directory_name}/test")
        self.model_saver = ModelSaverService(
            path=Path(f"{directory_name}"),
            topk=2,
            name=name,
        )
        self.train_report_rate = train_report_rate
        self.hyperparameter = hyperparameter

    def run_trainer(self, device: str):
        experiment = CaptchaExperiment(
            hyperparameter=self.hyperparameter,
            device=device,
        )

        train_dataloader = experiment["train_dataloader"]
        test_dataloader = experiment["test_dataloader"]
        model = experiment["model"]
        scheduler = experiment["scheduler"]
        preprocessor = experiment["preprocessor"]
        optimizer = experiment["optimizer"]

        self.train(
            epochs=self.hyperparameter.epoch,
            model=model,
            dataloader_train=train_dataloader,
            dataloader_test=test_dataloader,
            optimizer=optimizer,
            loss_fn=None,
            scheduler=scheduler,
            preprocess=preprocessor,
            device=device,
        )

    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        dataloader_train: DataLoader,
        dataloader_test: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        loss_fn,
        preprocess: v2.Compose,
        device: str,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}, ", end="")

            initial_time = time.time()
            self._train_one_epoch(
                epoch=epoch,
                model=model,
                dataloader=dataloader_train,
                optimizer=optimizer,
                preprocess=preprocess,
                device=device,
                dtype=dtype,
                loss_fn=None,
                scheduler=scheduler,
            )
            time_taken = time.time() - initial_time
            print(f"time_taken: {time_taken}s")

            if dataloader_test is not None:
                pass
                self._eval_one_epoch(
                    epoch=epoch,
                    model=model,
                    dataloader=dataloader_test,
                    preprocess=preprocess,
                    device=device,
                    loss_fn=None,
                    train_dataset_length=len(dataloader_train),
                    dtype=dtype,
                )
                # self._visualize_one_epoch(
                #     epoch=epoch,
                #     model=model,
                #     dataloader=dataloader_test,
                #     preprocess=preprocess,
                #     train_dataset_length=len(dataloader_train),
                #     device=device,
                # )
            self._save(model=model, epoch=epoch)

    def _save(self, model: torch.nn.Module, epoch: int):
        self.model_saver.save_without_shape(model, epoch)

    def _train_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        preprocess: v2.Compose,
        device: str,
        dtype,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        rate_to_print = max(math.floor(len(dataloader) * self.train_report_rate), 1)
        running_loss = 0.0

        scaler = torch.cuda.amp.grad_scaler.GradScaler()
        for index, data in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        ):
            with torch.autocast(device_type=device, dtype=dtype):
                inputs, labels = data
                outputs = model(inputs, labels, device=device)

            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer=optimizer)
            if scheduler is not None:
                scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            running_loss += outputs.loss.item()

            if index % rate_to_print == (rate_to_print - 1):
                current_training_sample = epoch * len(dataloader) + index + 1
                self.writer_train.add_scalar(
                    "loss",
                    running_loss / rate_to_print,
                    current_training_sample,
                )
                running_loss = 0.0

    def _eval_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        preprocess: v2.Compose,
        loss_fn,
        device: str,
        train_dataset_length: int,
        dtype,
    ):
        sum_loss = 0.0

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=dtype):
                for data in dataloader:
                    inputs, labels = data
                    outputs = model(inputs, labels, device)

        sum_loss += outputs.loss.item()
        iteration = (epoch + 1) * train_dataset_length
        avg_loss = sum_loss / len(dataloader)

        self.writer_test.add_scalar("loss", avg_loss, iteration)

    # def _visualize_one_epoch(
    #     self,
    #     epoch: int,
    #     model: torch.nn.Module,
    #     dataloader: DataLoader,
    #     device: Union[torch.device, str],
    #     preprocess: v2.Compose,
    #     train_dataset_length: int,
    # ):
    #     with torch.no_grad():
    #         for data in dataloader:
    #             inputs: torch.Tensor
    #             labels: torch.Tensor
    #             inputs, labels = data

    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             original_image = inputs
    #             inputs, labels = preprocess(inputs, labels)

    #             outputs = model(inputs)
    #             # colors = [
    #             #     (0, 0, 128),
    #             #     (128, 64, 128),
    #             #     (0, 128, 0),
    #             #     (0, 128, 128),
    #             #     (128, 0, 64),
    #             #     (192, 0, 192),
    #             #     (128, 0, 0),
    #             # ]

    #             visualization_image = generate_visualization(
    #                 original_image=original_image,
    #                 prediction=outputs,
    #                 target=labels,
    #             )

    #             # visualization_image = original_image[0]
    #             # for i in range(outputs.size(1) - 1):
    #             #     # Visualization for label
    #             #     visualization_image = draw_segmentation_masks(
    #             #         visualization_image,
    #             #         labels[0, i + 1] > 0.5,
    #             #         colors=colors[i],
    #             #         alpha=0.6,
    #             #     )
    #             #     # Visualization for prediction
    #             #     visualization_image = draw_segmentation_masks(
    #             #         visualization_image,
    #             #         outputs[0, i + 1].sigmoid() > 0.5,
    #             #         colors=colors[i],
    #             #         alpha=0.3,
    #             #     )

    #             iteration = (epoch + 1) * train_dataset_length
    #             self.writer_test.add_image(
    #                 tag="images",
    #                 img_tensor=visualization_image,
    #                 global_step=iteration,
    #             )
    #             break
