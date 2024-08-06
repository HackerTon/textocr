from datasets import load_metric
import torch


def dice_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon=1e-4,
):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    nominator = 2 * torch.sum(pred_flat * target_flat)
    denominator = torch.sum(pred_flat) + torch.sum(target_flat)
    return (nominator + epsilon) / (denominator + epsilon)


def dice_index_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon=1e-4,
):
    pred_flat = pred.permute([1, 0, 2, 3]).flatten(1)
    label_flat = target.permute([1, 0, 2, 3]).flatten(1)
    nominator = 2 * torch.sum(pred_flat * label_flat, dim=1)
    denominator = torch.sum(pred_flat, dim=1) + torch.sum(label_flat, dim=1)
    return (nominator + epsilon) / (denominator + epsilon)


def total_loss(pred: torch.Tensor, target: torch.Tensor):
    crossentropy_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target
    )
    dice_loss = 1 - dice_index(pred.sigmoid(), target)
    return crossentropy_loss + dice_loss


cer_metric = load_metric("cer")


def compute_cer(pred_ids, label_ids, processor):
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
