import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class BinaryTokenizer:
    def __init__(self):
        self.stoi = {"0": 0, "1": 1}
        self.itos = {0: "0", 1: "1"}

    def tokenize(self, binary_data):
        return [self.stoi[b] for b in binary_data]

    def detokenize(self, tokens):
        return "".join([self.itos[t] for t in tokens])


class NBitsTokenizer:
    def __init__(self, n_bits):
        self.target_bits = n_bits

    def _create_stoi_mapping(self, n_bits):
        max_val = 2**n_bits
        stoi = {"{:0{}b}".format(i, n_bits): i for i in range(max_val)}
        return stoi

    def tokenize(self, binary_data):
        assert (
            len(binary_data) % self.target_bits == 0
        ), "Length of binary_data must be multiple of n_bits."

        return [
            int(binary_data[i : i + self.target_bits], 2)
            for i in range(0, len(binary_data), self.target_bits)
        ]

    def detokenize(self, tokens):
        return "".join(
            [format(token, "0{}b".format(self.target_bits)) for token in tokens]
        )


class BinaryDataset(Dataset):
    def __init__(
        self, binary_data, config, start_index=0, end_index=None, is_training=False
    ):
        self.bintokenizer = BinaryTokenizer()
        self.text = "".join(f"{i:08b}" for i in binary_data[start_index:end_index])
        # we set the target_bits to 1 for IS_AUTOREGRESSIVE = True and to the desired value otherwise
        if is_training and config["is_autoregressive"]:
            self.target_bits = 1
        else:
            self.target_bits = config["target_bits"]
        self.seqlen = config["seqlen"]
        self.step = config["step"]
        self.batch_size = config["batch_size"]
        self.is_training = is_training

    def __len__(self):
        # Calculate the total number of full steps in the dataset
        total_steps = (len(self.text) - self.seqlen - self.target_bits) // self.step
        # Calculate the number of complete batches
        num_batches = total_steps // self.batch_size
        # Return the total number of samples in complete batches
        return num_batches * self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError

        start = idx * self.step
        # offset = self.target_bits - 1 if self.target_bits > 1 else 0
        offset = self.target_bits
        end = start + self.seqlen + offset
        binary_data = self.text[start:end]
        # We tokenize the inputs equally regardless of the target_bits
        tokens = self.bintokenizer.tokenize(binary_data)
        tokens = torch.tensor(tokens, dtype=torch.long)

        if self.target_bits == 1:
            tokens_one_hot = F.one_hot(tokens, num_classes=2).float()
            input_sequence = tokens[:-1]
            target_sequence = tokens_one_hot[1:]
        else:
            # Tokenize the binary data into bits
            tokens = [int(b) for b in binary_data]  # Convert each bit to an integer
            # tokens_tensor = torch.tensor(tokens, dtype=torch.long)
            tokens_tensor = torch.tensor(tokens, dtype=torch.int32)

            # Create the input sequence (all bits except the last 8)
            input_sequence = tokens_tensor[: -(self.target_bits)]
            target_sequence = []

            for i in range(len(tokens) - self.target_bits):
                # Get the integer value of the next n_bits
                next_bits = int(binary_data[i + 1 : i + 1 + self.target_bits], 2)
                target_sequence.append(next_bits)
            target_sequence = torch.tensor(target_sequence, dtype=torch.long)
            # One-hot encode the target tensor
            target_one_hot = F.one_hot(
                target_sequence, num_classes=2**self.target_bits
            )
            # Ensure the target is the correct shape: [sequence_length, num_classes]
            target_sequence = target_one_hot.view(-1, 2**self.target_bits)

        if len(input_sequence) != self.seqlen or (
            self.target_bits > 1 and len(target_sequence) != self.seqlen
        ):
            raise ValueError("Sequence length mismatch.")

        return input_sequence, target_sequence


def load_and_prepare_data(config):
    with open(config["filename"], "rb") as f:
        binary_data = f.read(config["num_bytes"])

    total_length = len(binary_data)
    train_length = int(total_length * config["train_ratio"])

    train_dataset = BinaryDataset(
        binary_data, config, end_index=train_length, is_training=True
    )
    eval_dataset = BinaryDataset(binary_data, config, start_index=train_length)

    return (
        DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        ),
        DataLoader(
            eval_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        ),
    )


if __name__ == "__main__":
    print("This is a module for loading and preparing data for the GPT-2 model.")
