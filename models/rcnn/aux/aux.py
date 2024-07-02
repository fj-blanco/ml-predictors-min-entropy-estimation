import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, utils_path)
from utils.nice_log import nice_log


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def get_available_gpu_memory():
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        try:
            device_name = gpu_devices[0].name.split(":")[-1]  # gets '0'
            device_name = f"GPU:{device_name}"  # constructs 'GPU:0'
            gpu_memory_info = tf.config.experimental.get_memory_info(device_name)
            available_memory = gpu_memory_info["current"]
            return available_memory
        except RuntimeError as e:
            print(e)
    return 0


def get_memory_usage():
    local_device_protos = device_lib.list_local_devices()
    for device in local_device_protos:
        if "GPU" in device.device_type:
            print(f"Device: {device.name}")
            print(f"Total Memory: {device.memory_limit/(1024*1024)} MB")
            print("-------------------------------------------------------")


def gpu_config(memory_fraction=0.5):
    # GPU config
    num_gpus = len(tf.config.list_physical_devices("GPU"))
    if num_gpus > 0:
        nice_log(f"Num GPUs Available: {num_gpus}")
        nice_log(f"Available devices: {get_available_devices()}")
        available_memory = get_available_gpu_memory()
        nice_log(f"Available GPU memory: {available_memory} bytes")
        get_memory_usage()
    else:
        nice_log("No GPUs available.")


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
):
    model_name = f"{model_name}_bits"
    total_elements = num_bytes

    WEIGHTS_DIR = f"./weights/{model_name}/{generator}"
    RESULTS_DIR = f"./results/{model_name}/{generator}"

    if "_mod" in filename:
        generator += filename.split("_mod")[1].split("-")[0]
    output_string_filename = f"{generator}_seqlen_{seqlen}_step_{step}_num_bytes_{num_bytes}_train_ratio_{train_ratio}_test_ratio_{test_ratio}"
    weights_filename = f"{output_string_filename}_weights.hdf5"
    second_model_weights_filename = (
        f"{output_string_filename}_second_model_weights.hdf5"
    )
    weights_path = os.path.join(WEIGHTS_DIR, weights_filename)
    second_model_weights_path = os.path.join(WEIGHTS_DIR, second_model_weights_filename)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    total_train_samples = int(
        total_elements * train_ratio
    )  # this adjusts the total samples for the training ratio
    steps_per_epoch = (
        total_train_samples // batch_size
    )  # this calculates the number of steps per epoch
    train_sequences = (
        int(total_elements * train_ratio) - int(np.ceil(seqlen / 8))
    ) // step
    validation_steps = int(test_ratio * total_elements // batch_size)

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
        "RESULTS_DIR": RESULTS_DIR,
        "output_string_filename": output_string_filename,
        "weights_path": weights_path,
        "second_model_weights_path": second_model_weights_path,
        "total_train_samples": train_sequences,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
    }


def log_model_parameters(model):
    total_parameters = model.count_params()
    trainable_parameters = np.sum(
        [tf.keras.backend.count_params(w) for w in model.trainable_weights]
    )
    non_trainable_parameters = np.sum(
        [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
    )

    nice_log(f"Total parameters: {total_parameters}")
    nice_log(f"Trainable parameters: {trainable_parameters}")
    nice_log(f"Non-trainable parameters: {non_trainable_parameters}")

    return total_parameters, trainable_parameters, non_trainable_parameters


def check_test_to_classes_ratio(train_ratio, test_ratio, config, target_bits):
    test_sequences = (test_ratio / train_ratio) * config["total_train_samples"]
    test_to_classes_ratio = test_sequences / (2**target_bits)
    nice_log(f"Test to classes ratio: {test_to_classes_ratio}")
    if test_to_classes_ratio < 1:
        raise ValueError("Test to classes ratio must be greater than 1")


if __name__ == "__main__":
    print("This is a module for auxiliary functions for the rcnn model.")
