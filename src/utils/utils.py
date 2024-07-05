import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_keypoints, draw_segmentation_masks


def combine_channels(image: torch.Tensor, colors: torch.Tensor, is_predict: bool):
    c, h, w = image.shape
    output_image = torch.zeros([h, w, 3], dtype=torch.uint8)
    for i in range(c):
        if is_predict:
            mask = image[i] > 0.5
        else:
            mask = image[i] == 1
        output_image[mask] = colors[i]
    return output_image


def visualize(
    input_image: torch.Tensor,
    grouth_truth: torch.Tensor,
    predicted: torch.Tensor,
):
    colors = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [128, 64, 128],
            [0, 128, 0],
            [128, 128, 0],
            [64, 0, 128],
            [192, 0, 192],
            [0, 0, 128],
        ],
        dtype=np.uint8,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=200)
    legend_patches = [
        patches.Patch(
            color=np.concatenate([color / 255, [1]]),
            label=UAVIDDataset.dataset_labels[idx],
        )
        for idx, color in enumerate(colors)
    ]
    fig.legend(handles=legend_patches, bbox_to_anchor=(1, 0.5))
    grouth_truth_image = combine_channels(grouth_truth, colors, False)
    predicted_image = combine_channels(predicted, colors, True)
    input_image = torch.permute(input_image[0], [1, 2, 0])

    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[2].set_axis_off()

    axes[0].set_title("Input Image")
    axes[1].set_title("Grouth Truth Image")
    axes[2].set_title("Predicted Image")

    axes[0].imshow(input_image)
    axes[1].imshow(grouth_truth_image)
    axes[2].imshow(predicted_image)


def generate_visualization(
    original_image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
):
    original_image = original_image.cpu()
    prediction = prediction.cpu()
    target = target.cpu()
    # Generate contours
    # and visualize as keypoints
    heart_mask = target[0].sigmoid()[2] > 0.5
    lung_mask = target[0].sigmoid()[1] > 0.5

    heart_mask = heart_mask.numpy().astype(np.uint8)
    lung_mask = lung_mask.numpy().astype(np.uint8)
    heart_contour, _ = cv2.findContours(
        heart_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    lung_contour, _ = cv2.findContours(
        lung_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    keypoints = torch.tensor(heart_contour)[..., 0, :]
    visualization_image = draw_keypoints(
        (original_image[0] * 255).to(torch.uint8),
        keypoints,
        radius=1,
        colors=(255, 0, 0),
    )
    for i in range(len(lung_contour)):
        keypoints = torch.tensor(lung_contour[i]).unsqueeze(0)[0, :]
        visualization_image = draw_keypoints(
            visualization_image,
            keypoints,
            radius=1,
            colors=(0, 255, 0),
        )

    # Visualize segmentation as mask
    output_heart_mask = prediction[0].sigmoid()[2] > 0.5
    output_lung_mask = prediction[0].sigmoid()[1] > 0.5

    visualization_image = draw_segmentation_masks(
        visualization_image,
        output_heart_mask,
        alpha=0.5,
        colors=(0, 0, 128),
    )
    visualization_image = draw_segmentation_masks(
        visualization_image,
        output_lung_mask,
        alpha=0.5,
        colors=(128, 64, 128),
    )

    return visualization_image
