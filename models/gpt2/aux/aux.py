import os
import sys
import torch

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, utils_path)
from utils.nice_log import nice_log


def binary_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    entropy = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
    return entropy.item()


def log_model_parameters(model):
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_parameters = total_parameters - trainable_parameters

    nice_log(f"Total parameters: {total_parameters}")
    nice_log(f"Trainable parameters: {trainable_parameters}")
    nice_log(f"Non-trainable parameters: {non_trainable_parameters}")
    return total_parameters, trainable_parameters, non_trainable_parameters


def get_config(
    model_name,
    filename,
    generator,
    seqlen,
    step,
    num_bytes,
    target_bits,
    train_ratio,
    test_ratio,
    learning_rate,
    batch_size,
    epochs,
    is_autoregressive,
    evaluate_all_bits,
):
    WEIGHTS_DIR = f"./weights/{model_name}/{generator}"
    RESULTS_DIR = f"./results/{model_name}/{generator}"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    output_string_filename = f"{generator}_sequence_{seqlen}_step_{step}_batch_{batch_size}_train_ratio_{train_ratio}_numbytes_{num_bytes}"
    weights_filename = f"{output_string_filename}.pth"
    weights_path = os.path.join(WEIGHTS_DIR, weights_filename)

    return {
        "filename": filename,
        "seqlen": seqlen,
        "step": step,
        "num_bytes": num_bytes,
        "target_bits": target_bits,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "is_autoregressive": is_autoregressive,
        "evaluate_all_bits": evaluate_all_bits,
        "WEIGHTS_DIR": WEIGHTS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "output_string_filename": output_string_filename,
        "weights_path": weights_path,
    }


if __name__ == "__main__":
    print("This is a module for auxiliary functions for the GPT-2 model.")
