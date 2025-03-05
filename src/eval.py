import torch
from aimet_torch.utils import in_eval_mode, get_device
from tqdm import tqdm
from src.data_loader import load_labeled_data , load_unlabeled_data
from src.logger import setup_logger


from typing import Optional
from src.config_loader import ConfigManager

config = ConfigManager.get_config()

logger = setup_logger(config["paths"]["log_file"], config["logging"]["log_level"])


def eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:
    """
    Evaluation callback for AutoQuant. Computes Top-1 and Top-5 accuracy.
    """
    _, dataloader = load_labeled_data(num_classes=config["general"]["num_classes"],images_per_class=config["general"]["images_per_class"], batch_size=config["general"]["batch_size"], is_train=True)
    device = get_device(model)

    correct_top1 = 0
    correct_top5 = 0
    total = 0  # Track total samples

    with in_eval_mode(model), torch.no_grad():
        for image, label in tqdm(dataloader, desc="Evaluating model"):
            image, label = image.to(device), label.to(device)

            logits = model(image)
            topk = logits.topk(k=5).indices  # Get top 5 predictions

            correct_top1 += (topk[:, 0] == label).sum().item()  # Top-1 matches
            correct_top5 += (topk == label.view(-1, 1)).sum().item()  # Top-5 matches
            total += label.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    logger.info(f"Top-1 Accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 Accuracy: {top5_acc:.4f}")

    return top1_acc  # AutoQuant requires a single accuracy metric
