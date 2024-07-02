# Standard library imports
import os
import sys
import time
from timeit import default_timer as timer

# Related third-party imports
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm

# Local application/library specific imports
from .argparser.argparser import parse_arguments
from .data_proc.data_proc import load_and_prepare_data, NBitsTokenizer
from .inference.inference import (
    autoregressive_inference,
    multitoken_inference,
    binary_inference,
)
from .aux.aux import binary_entropy, log_model_parameters, get_config

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, utils_path)
from utils.nice_log import nice_log

MODEL_NAME = "gpt2"


def save_model(model, filename):
    # Save model state
    torch.save(model.state_dict(), filename)


def build_model(
    config, n_positions=512, n_ctx=512, n_embd=768, n_layer=12, n_head=12, target_bits=1
):
    if config["is_autoregressive"]:
        vocab_size = 2
    else:
        vocab_size = 2**target_bits

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    model = GPT2LMHeadModel(config)

    # Trying parallelization:
    if torch.cuda.device_count() > 1:
        # DataParallel will divide and send input data to defined devices and combine them and produce output
        model = torch.nn.DataParallel(model)
    return model


def train_model(
    model,
    config,
    data,
    device,
    evaluation_checkpoints,
    eval_data,
    target_bits=1,
    accumulation_steps=4,
):
    bytes_processed = 0
    next_checkpoint_idx = 0
    partial_evals = []
    start = timer()
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    if config["is_autoregressive"]:
        # We train the model to predict the next bit in the sequence
        target_bits = 1
    if target_bits == 1:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    model.train()

    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        optimizer.zero_grad()  # reset gradients for the new epoch

        for i, (x, y) in enumerate(tqdm(data)):
            x = x.to(device)
            y = y.to(device)

            number_of_bytes_in_batch = (
                x.shape[0] * x.shape[1] // 8
            )  # x has shape (batch_size, seqlen - 1)
            bytes_processed += number_of_bytes_in_batch

            # Check if the accumulated size has exceeded the next checkpoint
            while (
                next_checkpoint_idx < len(evaluation_checkpoints)
                and bytes_processed >= evaluation_checkpoints[next_checkpoint_idx]
            ):
                eval_results = evaluate_model(
                    model, config, eval_data, device, target_bits=config["target_bits"]
                )
                # Creating a dictionary to store the evaluation results, bytes processed, and number of sequences
                checkpoint_data = {
                    "eval": eval_results,
                    "bytes_processed_eval": bytes_processed,
                }
                partial_evals.append(checkpoint_data)
                next_checkpoint_idx += 1
                time.sleep(60)

            if (
                torch.isnan(x).any()
                or torch.isinf(x).any()
                or torch.isnan(y).any()
                or torch.isinf(y).any()
            ):
                raise Exception("Invalid values found")

            with autocast():
                output = model(x)

                if target_bits == 1:
                    logits = output.logits
                    loss = loss_fn(logits, y)
                else:
                    # Flatten the logits to a 2D tensor where each row has num_classes columns
                    logits = output.logits.view(-1, 2**target_bits)
                    # Convert the target to a 1D tensor with class indices
                    _, target_indices = y.max(
                        dim=2
                    )  # This extracts the index of the max value in one-hot encoding
                    target = target_indices.view(
                        -1
                    )  # Flatten the target_indices to match logits' first dimension
                    # Compute the loss
                    loss = loss_fn(logits, target)

            # Scales loss, calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # Gradient accumulation part
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()

    training_time = float(timer() - start) / 60
    return training_time, partial_evals


def evaluate_model(model, config, data, device, target_bits=1):
    start = timer()
    model.eval()

    total = correct = 0
    total_cross_entropy = 0
    n_zeroes = 0
    all_binary_predictions = []

    if target_bits == 1:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        tokenizer = NBitsTokenizer(n_bits=target_bits)

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(data)):
            x = x.to(device)
            y = y.to(device)
            if target_bits == 1:
                binary_predictions, loss, correct, total = binary_inference(
                    model, x, y, loss_fn, correct, total
                )
            else:
                if config["is_autoregressive"]:
                    # Flatten the logits to match the target shape
                    binary_predictions, loss, correct, total = autoregressive_inference(
                        model,
                        x,
                        y,
                        target_bits,
                        loss_fn,
                        correct,
                        total,
                        config,
                        device,
                    )

                else:
                    binary_predictions, loss, correct, total = multitoken_inference(
                        model,
                        x,
                        y,
                        target_bits,
                        loss_fn,
                        correct,
                        total,
                        config,
                        tokenizer,
                    )

            all_binary_predictions.append(binary_predictions)

            # Counting zeroes
            n_zeroes += y[:, :, 0].sum().item()

            # Compute the cross-entropy loss for this mini-batch
            total_cross_entropy += loss.item()

    concatenated_binary_predictions = torch.cat(all_binary_predictions, dim=0)
    p_zeroes = (
        concatenated_binary_predictions == 0
    ).sum().float() / concatenated_binary_predictions.numel()
    overall_entropy = binary_entropy(p_zeroes)

    average_cross_entropy = total_cross_entropy / len(data)

    p_ml = correct / total
    p_g = 1 / (2**target_bits)
    p_c_zeroes = n_zeroes / (total // 2)
    p_c = max(p_c_zeroes, 1 - p_c_zeroes)

    results = {
        "training_time": float(timer() - start) / 60,
        "P_ML": p_ml,
        "P_g": p_g,
        "P_c": p_c,
        "P_e": overall_entropy,
        "bin_cross-entropy_loss": average_cross_entropy,
    }
    nice_log(
        f"EVALUATION RESULTS: Time taken: {float(timer()-start)/60:.1f} minutes, P_ML = {p_ml:.5f}, P_g= {p_g:.5f}, P_c = {p_c:.5f}, Predictions Entropy: {overall_entropy:.5f}, Binary Cross-Entropy Loss: {average_cross_entropy:.5f}"
    )

    return results


def main(
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
    model_size_parameters,
    evaluation_checkpoints=[],
):
    config = get_config(
        MODEL_NAME,
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
    )

    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    train_data, eval_data = load_and_prepare_data(config)
    steps_per_epoch = len(train_data)
    validation_steps = len(eval_data)

    # Build model
    model = build_model(config, **model_size_parameters, target_bits=target_bits)
    model_parameters = log_model_parameters(model)

    # Train model
    training_time, partial_evals = train_model(
        model,
        config,
        train_data,
        device,
        evaluation_checkpoints,
        eval_data,
        target_bits=target_bits,
    )
    # Save model
    save_model(model, config["weights_path"])
    # Final evaluation
    eval_results = evaluate_model(
        model, config, eval_data, device, target_bits=target_bits
    )
    final_checkpoint_data = {
        "eval": eval_results,
        "bytes_processed_eval": num_bytes,
    }

    partial_evals.append(final_checkpoint_data)

    total_train_samples = (
        int(num_bytes * train_ratio) - int(np.ceil(seqlen / 8))
    ) // step
    output_dict = {
        "training_time": "-" if training_time is None else f"{training_time:.1f}",
        "eval_results": partial_evals,
        "total_parameters": model_parameters[0],
        "trainable_parameters": model_parameters[1],
        "non_trainable_parameters": model_parameters[2],
        "total_train_samples": total_train_samples,
        "training_data_size": total_train_samples * seqlen,  # in bits
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
    }

    return output_dict


if __name__ == "__main__":
    args = parse_arguments()

    model_size_parameters = {
        "n_positions": args.seqlen,
        "n_ctx": args.seqlen,
        "n_embd": 768,
        "n_layer": 3,
        "n_head": 3,
    }

    results = main(
        args.filename,
        args.generator,
        args.seqlen,
        args.step,
        args.num_bytes,
        args.target_bits,
        args.train_ratio,
        args.test_ratio,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.is_autoregressive,
        args.evaluate_all_bits,
        model_size_parameters,
        evaluation_checkpoints=args.evaluation_checkpoints,
    )
