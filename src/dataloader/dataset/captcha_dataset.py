import boto3
import os
import torch
import orjson
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image


class CaptchaDataset(Dataset):
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.parsed_json = orjson.loads(open(json_path).read())
        self.s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )
        self.temp_directory = Path(".temp")
        self.temp_directory.mkdir(exist_ok=True)

        # Download all images minio
        self.load()

        # Check if all the files has been downloaded
        download_files_list = [x for x in self.temp_directory.glob("*.png")]
        if len(self.parsed_json) != len(download_files_list):
            raise Exception("S3 download incomplete")

    def load(self):
        for datum in self.parsed_json:
            s3_image_path = datum["data"]["captioning"]
            self.download(s3_image_path)

    def download(self, s3_path: str):
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts[0]
        object_name = "/".join(path_parts[1:])
        filename_path = self.temp_directory.joinpath(path_parts[-1]).absolute()

        # Skip download if file exist
        if filename_path.exists():
            return

        filename = str(filename_path)
        self.s3_client.download_file(bucket, object_name, filename)

    def get_file_local_from_s3(self, s3_path: str):
        path_parts = s3_path.replace("s3://", "").split("/")
        return str(self.temp_directory.joinpath(path_parts[-1]).absolute())

    def __len__(self):
        return len(self.parsed_json)

    def __getitem__(self, index):
        datum = self.parsed_json[index]
        result = datum["annotations"][0]["result"]
        if len(result) == 0:
            print(datum)
        label = result[0]["value"]["text"][0]
        s3_image_path = datum["data"]["captioning"]
        image = read_image(self.get_file_local_from_s3(s3_image_path))
        return image, label


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset = CaptchaDataset(
        "/pool/storage/projects/textocr/project-2-at-2024-08-06-07-03-5a7af7f3.json"
    )
    print(len(dataset))
    for image, label in dataset:
        print(label)
