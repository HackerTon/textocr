import torch
from torchvision.transforms import Normalize

from model.model import BackboneType, MultiNet
from torchvision.io import read_image, ImageReadMode, write_png
from torchvision.transforms.functional import resize, InterpolationMode
from utils.utils import combine_channels
import cv2


def main():
    if torch.backends.mps.is_available():
        print("Using MPS engine")
        device = "mps"
    elif torch.cuda.is_available():
        print("Using CUDA engine")
        device = "cuda"
    else:
        print("Using CPU engine")
        device = "cpu"

    model = MultiNet(numberClass=3, backboneType=BackboneType.RESNET50)
    preprocessor = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

    # Load weights into model
    model.load_state_dict(
        torch.load(
            "data/model/cardiac_model_25.pt",
            map_location=device,
        )
    )

    # Load image
    file_path = "data/cardiac/chestxray/images_001/images/00000001_000.png"
    image = read_image(file_path, ImageReadMode.RGB)
    image = (image / 255).float().unsqueeze(0)

    model.eval()
    with torch.no_grad():
        image = resize(image, [512, 512], interpolation=InterpolationMode.NEAREST)
        image = preprocessor(image)
        output_tensor = model(image)
        visualization_tensor = generate_visualization(output_tensor)

        write_png(visualization_tensor, "visualization.png")


def generate_ctr(output_tensor: torch.Tensor):
    output_tensor = torch.clone(output_tensor).cpu()


def generate_visualization(outputs: torch.Tensor):
    colors = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 128],
            [128, 64, 128],
        ],
        dtype=torch.uint8,
    )
    predicted_image = combine_channels(outputs[0], colors, True)
    predicted_image = predicted_image[..., [2, 1, 0]].permute([2, 0, 1]).cpu()
    return predicted_image


if __name__ == "__main__":
    main()
