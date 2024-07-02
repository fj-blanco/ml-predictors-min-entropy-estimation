# Standard library imports
import os
import sys
import gc
import time
from timeit import default_timer as timer

# Related third-party imports
import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Convolution1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy

# Local application/library specific imports
from .argparser.argparser import parse_arguments
from .aux.aux import (
    get_config,
    gpu_config,
    get_memory_usage,
    log_model_parameters,
    check_test_to_classes_ratio,
)
from .data_proc.data_proc import data_generator

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, utils_path)
from utils.nice_log import nice_log

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
MODEL_NAME = "rcnn"

# Get the current process
process = psutil.Process()


def build_model(config, target_bits=1, scale_factor=1):
    model = Sequential()
    dim = 2
    loss = "binary_crossentropy"
    output_dim = 2**target_bits

    if scale_factor == 1:
        model.add(
            Convolution1D(
                filters=64,
                kernel_size=12,
                padding="same",
                activation="relu",
                input_shape=(config["seqlen"], dim),
            )
        )
        model.add(
            Convolution1D(filters=128, kernel_size=3, padding="same", activation="relu")
        )
    elif scale_factor == 2:
        model.add(
            Convolution1D(
                filters=32,
                kernel_size=12,
                padding="same",
                activation="relu",
                input_shape=(config["seqlen"], dim),
            )
        )
        model.add(
            Convolution1D(filters=64, kernel_size=6, padding="same", activation="relu")
        )
        model.add(
            Convolution1D(filters=128, kernel_size=3, padding="same", activation="relu")
        )
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
    else:
        raise ValueError("Unknown scale factor.")

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    model.compile()
    return model


def binary_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def evaluate_model(model, config):
    start = timer()

    eval_gen = data_generator(config, start=config["train_ratio"])
    n_true = 0
    n_zeroes = 0
    n_total = 0
    interval = 10000  # interval to monitor system usage
    count = 0  # counter for the monitoring interval
    all_binary_predictions = []

    total_cross_entropy = 0
    loss_fn = BinaryCrossentropy(from_logits=False)

    for i in range(config["validation_steps"]):
        start_loop = timer()
        Xt, yt = next(eval_gen)
        # Convert the numpy array Xt to tensor
        Xt_tensor = tf.convert_to_tensor(Xt, dtype=tf.bool)
        preds = model(Xt_tensor)
        predictions = np.argmax(preds, axis=-1)
        evaluation_data = np.argmax(yt, axis=-1)
        all_binary_predictions.append(predictions)
        n_true += np.sum(predictions == evaluation_data)
        n_zeroes += np.sum(evaluation_data == 0)
        n_total += preds.shape[0]
        cross_entropy = loss_fn(yt, preds)
        total_cross_entropy += cross_entropy.numpy()

        # monitor system usage every 'interval' steps
        if count % interval == 0:
            percentage_elapsed = (i + 1) / config["validation_steps"] * 100
            cpu_percentages = psutil.cpu_percent(interval=1.0, percpu=True)
            nice_log(f"Step: {count}")
            nice_log(f"Percentage elapsed: {percentage_elapsed:.2f}")
            nice_log(
                f"Average CPU usage across cores: {(sum(cpu_percentages) / len(cpu_percentages)):.2f}"
            )
            nice_log(
                f"Memory usage (GB): {(process.memory_info().rss / (1024 ** 3)):.2f}"
            )
            nice_log(f"Number of threads: {process.num_threads()}")
            nice_log(f"Time elapsed (seconds): {timer()- start_loop:.1f} seconds")

        del preds
        _ = gc.collect()

        count += 1

    p_ml = n_true / n_total
    p_g = 1 / 2 ** config["target_bits"]
    p_c = max(n_zeroes / n_total, 1 - (n_zeroes / n_total))

    concatenated_binary_predictions = np.concatenate(all_binary_predictions, axis=0)
    p_zeroes = (
        np.sum(concatenated_binary_predictions == 0)
        / concatenated_binary_predictions.size
    )
    overall_entropy = binary_entropy(p_zeroes)

    average_cross_entropy = total_cross_entropy / config["validation_steps"]

    results = {
        "training_time": float(timer() - start) / 60,
        "P_ML": p_ml,
        "P_g": p_g,
        "P_c": p_c,
        "P_e": overall_entropy,
        "bin_cross-entropy_loss": average_cross_entropy,
    }

    nice_log(
        f"EVALUATION RESULTS: Time taken: {float(timer()-start)/60:.1f} minutes, P_ML = {p_ml:.5f}, P_g = {p_g:.5f}, P_c = {p_c:.5f},"
        f"Predictions Entropy = {overall_entropy:.5f}, Binary Cross-Entropy Loss = {average_cross_entropy:.5f}"
    )
    return results


def indices_to_one_hot_tf(predicted_bits, num_bits):
    def format_binary_string(idx, num_bits):
        return format(idx, "0{}b".format(num_bits))

    # Create a tensor to store one-hot encoded sequences
    one_hot_sequences = tf.TensorArray(dtype=tf.float32, size=len(predicted_bits))

    for i in tf.range(len(predicted_bits)):
        idx = predicted_bits[i]
        bit_sequence = tf.py_function(
            func=format_binary_string, inp=[idx, num_bits], Tout=tf.string
        )
        bit_sequence = tf.strings.bytes_split(bit_sequence)
        bit_sequence = tf.strings.to_number(bit_sequence, out_type=tf.float32)
        one_hot_bit_sequence = tf.one_hot(tf.cast(bit_sequence, tf.int32), depth=2)
        one_hot_sequences = one_hot_sequences.write(i, one_hot_bit_sequence)

    one_hot_sequences = one_hot_sequences.stack()
    one_hot_sequences = tf.cast(one_hot_sequences, tf.bool)

    return one_hot_sequences


def train_model(model, config, evaluation_checkpoints, first_model=None):
    start = timer()

    # Define other necessary variables
    bytes_processed = 0
    next_checkpoint_idx = 0
    partial_evals = []

    dim = 2

    if config["target_bits"] == 1:
        output_dim = dim
        output_type = bool
    else:
        output_dim = 2 ** config["target_bits"]
        output_type = bool
    # Define generator for training
    train_gen = data_generator(
        config,
        epochs=config["epochs"],
        end=config["train_ratio"],
    )

    # Define dataset for training
    train_dataset = tf.data.Dataset.from_generator(
        generator=lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, config["seqlen"], dim), dtype=bool),
            tf.TensorSpec(shape=(None, output_dim), dtype=output_type),
        ),
    )

    # Define loss and optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=config["learning_rate"])

    for epoch in range(config["epochs"]):
        for i, (x, y) in enumerate(train_dataset):
            # Incrementing the samples processed
            bytes_processed += (x.shape[0] * x.shape[1]) // 8

            # Check if the accumulated size has exceeded the next checkpoint
            while (
                next_checkpoint_idx < len(evaluation_checkpoints)
                and bytes_processed >= evaluation_checkpoints[next_checkpoint_idx]
            ):
                eval_results = evaluate_model(model, config)
                checkpoint_data = {
                    "eval": eval_results,
                    "bytes_processed_eval": bytes_processed,
                }
                partial_evals.append(checkpoint_data)
                next_checkpoint_idx += 1
                time.sleep(180)

            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss_fn(y, logits)
                # Compute gradients
                grads = tape.gradient(loss_value, model.trainable_variables)
                # Update weights
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # Compute gradients
                grads = tape.gradient(loss_value, model.trainable_variables)
                # Update weights
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

    training_time = float(timer() - start) / 60
    nice_log(f"Training time: {training_time:.1f} minutes")
    return model, training_time, partial_evals


def evaluate_and_record(model, config, num_bytes, partial_evals):
    eval_results = evaluate_model(model, config)
    final_checkpoint_data = {
        "eval": eval_results,
        "bytes_processed_eval": num_bytes,
    }
    partial_evals.append(final_checkpoint_data)
    return partial_evals


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
    model_size_parameters,
    evaluation_checkpoints=[],
):
    gpu_config()

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
    )

    check_test_to_classes_ratio(train_ratio, test_ratio, config, target_bits)

    model = build_model(config, **model_size_parameters, target_bits=target_bits)
    model_parameters = log_model_parameters(model)

    model, training_time, partial_evals = train_model(
        model,
        config,
        evaluation_checkpoints,
    )
    # Save weights after training
    model.save_weights(config["weights_path"])

    nice_log("Evaluating first model...")
    partial_evals = evaluate_and_record(model, config, num_bytes, partial_evals)

    output_dict = {
        "training_time": "-" if training_time is None else f"{training_time:.1f}",
        "eval_results": partial_evals,
        "total_parameters": model_parameters[0],
        "trainable_parameters": model_parameters[1],
        "non_trainable_parameters": model_parameters[2],
        "total_train_samples": config["total_train_samples"],
        "training_data_size": (config["total_train_samples"]) * seqlen,  # in bits
        "steps_per_epoch": config["steps_per_epoch"],
        "validation_steps": config["validation_steps"],
    }
    return output_dict


if __name__ == "__main__":
    args = parse_arguments()

    nice_log("Arguments: {}".format(args))

    model_size_parameters = dict(scale_factor=2)

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
        model_size_parameters,
        evaluation_checkpoints=args.evaluation_checkpoints,
    )
