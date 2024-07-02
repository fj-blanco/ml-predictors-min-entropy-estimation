import os
import sys
import gc
import mmap

import numpy as np

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, utils_path)


def bytes_to_bits(bytes_arr):
    np_arr = np.frombuffer(
        bytes_arr, dtype=np.uint8
    )  # This converts bytes to numpy array
    return np.unpackbits(np_arr)


def data_generator(config, epochs=1, start=0.0, end=1.0):
    epoch_count = 0
    while epoch_count < epochs:
        with open(config["filename"], "rb") as file:
            mapped_file = mmap.mmap(
                file.fileno(), length=config["num_bytes"], access=mmap.ACCESS_READ
            )
            file_len = mapped_file.size()
            # Raise error if num_bytes is larger than file size
            if file_len < config["num_bytes"]:
                raise ValueError("num_bytes is larger than file size")

            start_pos = int(start * config["num_bytes"])
            end_pos = int(end * config["num_bytes"])

            number_of_batches = len(range(start_pos, end_pos, config["batch_size"]))

            for batch_start in range(start_pos, end_pos, config["batch_size"]):
                batch_end = min(batch_start + config["batch_size"], end_pos)

                mapped_data = bytes_to_bits(mapped_file[batch_start:batch_end])
                bit_or_byte = 2  # 2 for bit values 0 and 1
                data_type = bool

                n = (len(mapped_data) - config["seqlen"]) // config[
                    "step"
                ]  # Number of samples
                X = np.zeros((n, config["seqlen"], bit_or_byte), dtype=data_type)
                if config["target_bits"] == 1:
                    y = np.zeros((n, bit_or_byte), dtype=data_type)
                    # One-hot encoding:
                    for i in range(n):
                        for t in range(config["seqlen"]):
                            index = mapped_data[i * config["step"] + t]
                            X[i, t, index] = 1
                        # this is the index of the last bit of the sequence
                        next_bits_start = i * config["step"] + config["seqlen"]
                        index = mapped_data[next_bits_start]
                        y[i, index] = 1
                else:
                    y = np.zeros((n, 2 ** config["target_bits"]), dtype=data_type)

                    # Now we get the next target_bits bits instead the next bit
                    for i in range(n):
                        for t in range(config["seqlen"] - config["target_bits"]):
                            index = mapped_data[i * config["step"] + t]
                            X[i, t, index] = 1

                        # Next bits for y, starting just after the end of the current sequence
                        next_bits_start = (
                            i * config["step"]
                            + config["seqlen"]
                            - config["target_bits"]
                        )
                        next_bits_end = next_bits_start + config["target_bits"]
                        next_bits = mapped_data[next_bits_start:next_bits_end]
                        next_bits = "".join(str(bit) for bit in next_bits)
                        y_index = int(next_bits, 2)
                        y[i, y_index] = 1  # One-hot encoding for y

                del mapped_data
                gc.collect()

                yield X, y

        epoch_count += 1


if __name__ == "__main__":
    print("This is a module for loading and preparing data for the rcnn model.")
