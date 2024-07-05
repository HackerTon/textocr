import json
from os import remove
from pathlib import Path

import torch
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


device = torch.device("cuda")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
ocrModel = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
ocrModel = ocrModel.to(device)

ocrModel.config.decoder_start_token_id = processor.tokenizer.cls_token_id
ocrModel.config.pad_token_id = processor.tokenizer.pad_token_id
ocrModel.config.eos_token_id = processor.tokenizer.sep_token_id
ocrModel.config.max_length = 10
ocrModel.config.no_repeat_ngram_size = 3
ocrModel.config.length_penalty = 2.0
ocrModel.config.num_beams = 4
ocrModel.config.max_new_tokens

optimizer = torch.optim.AdamW(ocrModel.parameters(), lr=1e-5)
modelSaver = ModelSaverService(path=Path("data/model"), topk=2)

training_data = ContainerOCRDatasetText(
    directory="./container_dataset/",
    processor=processor,
    is_train=True,
)

train_dataloader = DataLoader(
    training_data,
    batch_size=1,
    shuffle=True,
)

test_data = ContainerOCRDatasetText(
    directory="./container_dataset/",
    processor=processor,
    is_train=False,
)

test_dataloader = DataLoader(
    test_data,
    batch_size=5,
)

cer_metric = load_metric("cer")


def compute_cer(pred_ids, label_ids):
    sum_cer = 0
    for pred, label in zip(pred_ids, label_ids):
        pred_str = processor.decode(pred, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.decode(label, skip_special_tokens=True)

        if pred_str == "":
            sum_cer += len(label_str)
        elif label_str == "":
            sum_cer += len(pred_str)
        else:
            sum_cer += cer_metric.compute(
                predictions=[pred_str], references=[label_str]
            )
    return sum_cer / len(pred_ids)


for epoch in range(0, 200):
    train_running_loss = 0.0
    ocrModel.train()
    for data in tqdm(train_dataloader):
        inputs: torch.Tensor
        labels: torch.Tensor
        inputs, labels = data[0], data[1]

        inputs = inputs.to(device)
        labels = labels.to(device)

        output = ocrModel(inputs, labels=labels)
        output.loss.backward()
        train_running_loss += output.loss.item()
        optimizer.step()
        optimizer.zero_grad()

    print(
        f"Epoch: {epoch}, Running loss: {train_running_loss / len(train_dataloader)},",
        end="",
    )

    validation_cer = 0
    ocrModel.eval()
    for data in test_dataloader:
        with torch.no_grad():
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data[0], data[1]

            inputs = inputs.to(device)
            labels = labels.to(device)

            generated_ids = ocrModel.generate(inputs)
            validation_cer += compute_cer(generated_ids, labels)

    print(f"Running validation: {validation_cer / len(test_dataloader)}")
    modelSaver.save(ocrModel, epoch)
