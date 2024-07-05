import time

import torch
from torchvision.models import resnet34
from torchvision.transforms import Normalize

from src.model.model import BackboneType, MultiNet


def standard_resnet_benchmark():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = resnet34()
    model = model.to(device)

    print("Start Benchmark!")
    print(f"Running on {device}")
    BASELINE_BATCH = 32
    STRESS_BATCH = 32

    initial_time = time.time()
    for _ in range(100):
        random_sample = torch.randn([BASELINE_BATCH, 3, 512, 512], device=device)
        model(random_sample)
    baseline_diff = time.time() - initial_time

    initial_time = time.time()
    if device == "cuda":
        with torch.autocast(device, dtype=torch.float16):
            for _ in range(100):
                random_sample = torch.randn(
                    [STRESS_BATCH, 3, 512, 512, 3], device=device
                )
                model(random_sample)
        stressed_diff = time.time() - initial_time
    else:
        for _ in range(100):
            random_sample = torch.randn([STRESS_BATCH, 3, 512, 512], device=device)
            model(random_sample)
        stressed_diff = time.time() - initial_time

    print(baseline_diff / BASELINE_BATCH)
    print(stressed_diff / STRESS_BATCH)


def run_benchmark():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = MultiNet(numberClass=3, backboneType=BackboneType.RESNET50)
    preprocess = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = model.to(device)
    preprocess = preprocess.to(device)

    print("Start Benchmark!")
    print(f"Running on {device}")
    BASELINE_BATCH = 32
    STRESS_BATCH = 32

    initial_time = time.time()
    for _ in range(100):
        random_sample = torch.randn([BASELINE_BATCH, 3, 512, 512], device=device)
        model(random_sample)
    baseline_diff = time.time() - initial_time

    initial_time = time.time()
    if device == "cuda":
        with torch.autocast(device, dtype=torch.float16):
            for _ in range(100):
                random_sample = torch.randn([STRESS_BATCH, 3, 512, 512], device=device)
                model(random_sample)
        stressed_diff = time.time() - initial_time
    else:
        for _ in range(100):
            random_sample = torch.randn([STRESS_BATCH, 3, 512, 512], device=device)
            model(random_sample)
        stressed_diff = time.time() - initial_time

    print(baseline_diff / BASELINE_BATCH)
    print(stressed_diff / STRESS_BATCH)


if __name__ == "__main__":
    standard_resnet_benchmark()
