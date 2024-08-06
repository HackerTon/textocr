import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from src.service.hyperparamater import Hyperparameter
from src.service.trainer import Trainer

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(
    name: str,
    epoch: int,
    device: str,
    batch_size: int,
    path: str,
    learning_rate: float,
):
    if not Path(path).exists():
        print(f"Dataset not found in '{path}'")
        return

    hyperparameter = Hyperparameter(
        epoch=epoch,
        learning_rate=learning_rate,
        batch_size_test=16,
        batch_size_train=batch_size,
        data_path=path,
    )
    trainer = Trainer(
        train_report_rate=0.1,
        name=name,
        hyperparameter=hyperparameter,
    )
    trainer.run_trainer(device=device)


if __name__ == "__main__":
    load_dotenv()

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", default=50, type=int)
    parser.add_argument("-m", "--mode", default="cpu", type=str)
    parser.add_argument("-b", "--batchsize", default=1, type=int)
    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-l", "--learning_rate", default=0.001, type=float)
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name of the experiment",
    )
    parsed_data = parser.parse_args()
    run(
        name=parsed_data.name,
        epoch=parsed_data.epoch,
        device=parsed_data.mode,
        batch_size=parsed_data.batchsize,
        path=parsed_data.path,
        learning_rate=parsed_data.learning_rate,
    )
