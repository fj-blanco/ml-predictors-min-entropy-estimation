import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, help="File name")
    parser.add_argument("--generator", type=str, required=True, help="Generator")
    parser.add_argument("--seqlen", type=int, default=100, help="Max length in bits")
    parser.add_argument("--step", type=int, default=3, help="Step")
    parser.add_argument("--num_bytes", type=int, default=1, help="Data Length")
    parser.add_argument("--target_bits", type=int, default=1, help="Target bits")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train Ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test Ratio")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="Batch size"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--is_autoregressive",
        action="store_true",
        help="Enable autoregressive mode (default: False)",
    )
    parser.add_argument(
        "--evaluate_all_bits",
        action="store_true",
        help="Evaluate all bits (default: False)",
    )
    parser.add_argument(
        "--evaluation_checkpoints",
        type=int,
        nargs="*",
        default=[],
        help="Data size goals for evaluations",
    )
    args = parser.parse_args()

    if (
        args.train_ratio + args.test_ratio != 1
        or args.train_ratio < 0
        or args.train_ratio > 1
        or args.test_ratio < 0
        or args.test_ratio > 1
    ):
        raise ValueError(
            "train_ratio and test_ratio must be between 0 and 1 and their sum must be 1"
        )

    if args.target_bits < 1 or (args.target_bits > args.seqlen - 1):
        raise ValueError("target_bits must be between 1 and seqlen - 1")

    return args


if __name__ == "__main__":
    print("This is a module for parsing arguments for the GPT-2 model.")
