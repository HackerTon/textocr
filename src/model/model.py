import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List


class TextOcrModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        )

        self.ocr_model.config.decoder_start_token_id = (
            self.processor.tokenizer.cls_token_id
        )
        self.ocr_model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.ocr_model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.ocr_model.config.max_length = 10
        self.ocr_model.config.no_repeat_ngram_size = 3
        self.ocr_model.config.length_penalty = 2.0
        self.ocr_model.config.num_beams = 4

    def forward(self, image: torch.Tensor, text: List[str], device):
        """
        Return the loss value
        image: RGB, BCHW, range [0, 255], uint8
        text: BE, E = encoding dimension
        device: device, str
        """
        image_feature = torch.tensor(np.array(self.processor(image).pixel_values)).to(
            device
        )
        label_feature = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=15,
        ).input_ids
        for idx, labels in enumerate(label_feature):
            label_feature[idx] = [
                label if label != self.processor.tokenizer.pad_token_id else -100
                for label in labels
            ]
        label_feature = torch.tensor(np.array(label_feature)).to(device)
        loss = self.ocr_model(image_feature, labels=label_feature)
        return loss

    def generate(self, image: torch.Tensor, device):
        """
        Return generated text ids
        image: RGB, BCHW, range [0, 255], uint8
        device: device, str
        """
        image_feature = self.processor(image).to(device)
        return self.ocr_model.generate(image_feature)


if __name__ == "__main__":
    model = TextOcrModel()
    model.eval()
    with torch.no_grad():
        output = model(torch.rand([1, 3, 384, 384]), ["asdf"])
    print(output.loss)
