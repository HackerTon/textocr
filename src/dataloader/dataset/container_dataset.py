import json
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io.image import ImageReadMode, read_image
from torchvision.transforms.functional import rotate


class ContainerOCRDatasetText(Dataset):
    def __init__(self, directory_path):
        self.directory = Path(directory_path)
        self.image_label = []
        self.decode(
            file_path=str(self.directory.joinpath("train/_annotations.coco.json")),
            type="train",
        )
        self.decode(
            file_path=str(self.directory.joinpath("valid/_annotations.coco.json")),
            type="valid",
        )

    def decode(self, file_path: str, type: str):
        with open(file_path) as file:
            jsonData = json.load(file)
            for image in jsonData["images"]:
                image_id = image["id"]
                image_filename = image["file_name"]
                for annotation in jsonData["annotations"]:
                    if annotation["image_id"] == image_id:
                        bounding_box = annotation["bbox"]
                        x1, y1 = int(bounding_box[0]), int(bounding_box[1])
                        x2, y2 = x1 + int(bounding_box[2]), y1 + int(bounding_box[3])
                        self.image_label.append(
                            {
                                "image_filename": f"{type}/{image_filename}",
                                "bbox": [x1, y1, x2, y2],
                            }
                        )

    def __len__(self):
        return len(self.image_label)

    def decode_image(self, image_path):
        return read_image(image_path, ImageReadMode.RGB)

    def __getitem__(self, index):
        image_path = f"{self.image_label[index]['image_filename']}"
        image = self.decode_image(str(self.directory.joinpath(image_path)))
        text = self.image_label[index]["image_filename"].split("_")[1]
        x1, y1, x2, y2 = self.image_label[index]["bbox"]
        original_image = image[..., y1:y2, x1:x2]

        h, w = abs(y1 - y2), abs(x1 - x2)
        if h > w:
            original_image = rotate(original_image, angle=90, expand=True)
        return original_image, text


if __name__ == "__main__":
    dataset = ContainerOCRDatasetText(directory_path="data/container_dataset")
    for image, text in dataset:
        print(image.shape)
        print(text)
        break
