import os
import csv
import time
import argparse
import itertools
import subprocess
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from utils.nice_log import nice_log
import autoregressive_process.autoregressive_process as arp
from parsers.entropy_parsers import parse_entropy_output

OUTPUT_FILE_PATH = "./results"
ENTROPY_TEST_BINARY = "./SP800-90B_EntropyAssessment/cpp/ea_non_iid"
MIN_EVAL_ORDER = 4
EVALS_PER_ORDER = 2


def run_entropy_assessment(test_binary, file):
    command = [test_binary, "-a", "-v", file]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    print(proc.stdout)
    entropies_dict = parse_entropy_output(proc.stdout)
    return entropies_dict


def calculate_p_c(random_bytes, num_bytes=10**4):
    # Convert the random bytes to a bit array
    random_bits = np.unpackbits(np.frombuffer(random_bytes[:num_bytes], dtype=np.uint8))
    # Compute the number of zeroes
    n_zeroes = np.sum(random_bits == 0)
    # Compute p_c for the random bytes
    total_bits = len(random_bits)
    p_c_zeroes = n_zeroes / total_bits
    p_c_random_bytes = max(p_c_zeroes, 1 - p_c_zeroes)

    return p_c_random_bytes


def experimental_min_entropy(p_ml, target_bits=1):
    if target_bits == 1:
        min_entropy = -np.log2(p_ml)
    else:
        min_entropy = -np.log2(p_ml) / target_bits
    return min_entropy


def create_constant_dict(
    model_name,
    model_param_dict,
    hardware,
    corr_intensity,
    data_param_dict,
    p_c_random_bytes,
    min_entropy_th,
    entropies_dict,
):
    nn_info_unit = "bit"

    return {
        # Model parameters
        "model": model_name,
        "nn_info_unit": nn_info_unit,
        "hardware": hardware,
        "is_autoregressive": model_param_dict["is_autoregressive"],
        "evaluate_all_bits": model_param_dict["evaluate_all_bits"],
        # Data parameters
        "num_bytes": f'{model_param_dict["num_bytes"]}',
        "target_bits": f'{model_param_dict["target_bits"]}',
        "seqlen": f'{model_param_dict["seqlen"]}',
        "step": f'{model_param_dict["step"]}',
        "train_ratio": f'{model_param_dict["train_ratio"]}',
        "learning_rate": f'{model_param_dict["learning_rate"]}',
        "batch_size": f'{model_param_dict["batch_size"]}',
        "epochs": f'{model_param_dict["epochs"]}',
        "corr_intensity": f"{corr_intensity:.3f}",
        "autocorrelation_function": f'{data_param_dict["autocorrelation_function"]}',
        "distance_scale_p": data_param_dict["distance_scale_p"],
        "exponential_decay_rate": data_param_dict["exponential_decay_rate"],
        "gaussian_sigma": data_param_dict["gaussian_sigma"],
        "p_c_max": p_c_random_bytes,
        "min_entropy_th": min_entropy_th,
        # Entropies dict:
        **entropies_dict,
    }


def write_results_to_csv(output_dict, results_dir):
    output_file = f"{results_dir}/results.csv"
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(output_dict.keys())
        writer.writerow(output_dict.values())


def save_random_data(data, data_target_file, sample_target_file, sample_size=10**7):
    """
    We save the random data.
    We also save a sample of at most sample_size bytes (defaults to 100**7) to a file to be used by the NIST entropy assessment.
    """
    with open(data_target_file, "wb") as f:
        f.write(data)
    sample_data = data[:sample_size]
    with open(sample_target_file, "wb") as f:
        f.write(sample_data)


def check_test_to_classes_ratio(
    num_bytes, test_ratio, train_ratio, target_bits, seqlen, step
):
    threshold_ratio = 1
    total_elements = (
        num_bytes  # In bits mode, the total elements are simply the number of bytes.
    )

    train_sequences = (
        int(total_elements * train_ratio) - int(np.ceil(seqlen / 8))
    ) // step
    test_sequences = (test_ratio / train_ratio) * train_sequences
    test_to_classes_ratio = test_sequences / (2**target_bits)

    if test_to_classes_ratio < threshold_ratio:
        required_num_bytes = threshold_ratio * (2**target_bits) * step * (
            train_ratio / test_ratio
        ) + int(np.ceil(seqlen / 8))
        required_num_bytes = int(np.ceil(required_num_bytes))
        raise ValueError(
            f"Test to classes ratio must be greater than 1. Increase the number of bytes to at least {required_num_bytes}."
        )


def generate_gbAR_random_bytes(alpha_scaling_factor, data_param_dict, beta, num_bytes):
    distance_scale_p = data_param_dict["distance_scale_p"]
    autocorrelation_function = data_param_dict["autocorrelation_function"]
    if autocorrelation_function == "point-to-point":
        alpha = arp.point_to_point_alpha(distance_scale_p, alpha_scaling_factor)
    elif autocorrelation_function == "constant":
        alpha = arp.constant_alpha(distance_scale_p, alpha_scaling_factor)
    elif autocorrelation_function == "exponential":
        alpha = arp.exponentially_decreasing_alpha(
            distance_scale_p,
            alpha_scaling_factor,
            decay_rate=data_param_dict["exponential_decay_rate"],
        )
    elif autocorrelation_function == "gaussian":
        alpha = arp.gaussian_alpha(
            distance_scale_p,
            alpha_scaling_factor,
            data_param_dict["gaussian_sigma"],
            0.001,
        )
    # if it contains the substring constant_
    elif "constant_" in autocorrelation_function:
        signs = data_param_dict["signs"]
        alpha = arp.constant_alpha(distance_scale_p, alpha_scaling_factor, signs=signs)
    else:
        raise ValueError("Unknown autocorrelation function.")

    assert beta >= 0
    assert np.sum(np.abs(alpha)) + beta - 1 < 1e-10
    return arp.gbAR(alpha, beta, num_bytes), alpha


def generate_evaluation_checkpoints(start_order, end_order, num_points_per_order=2):
    evaluation_checkpoints = np.concatenate(
        [
            np.logspace(
                order, order + 1, num=num_points_per_order, endpoint=False
            ).astype(int)
            for order in range(start_order, end_order)
        ]
    )

    return evaluation_checkpoints


def execute_model(model_name, model_param_dict):
    # Dynamical imports to avoid conflict between torch and tensorflow
    if model_name == "gpt2":
        from models.gpt2 import rng_gpt2 as model

        if model_param_dict["batch_size"] is None:
            model_param_dict["batch_size"] = 8

        model_param_dict["model_size_parameters"] = {
            "n_positions": model_param_dict["seqlen"],
            "n_ctx": model_param_dict["seqlen"],
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
        }
    elif model_name == "rcnn":
        model_param_dict.pop("is_autoregressive", None)
        model_param_dict.pop("evaluate_all_bits", None)
        from models.rcnn import rng_rcnn as model

        if model_param_dict["batch_size"] is None:
            model_param_dict["batch_size"] = 2 * 10**3

        model_param_dict["model_size_parameters"] = dict(scale_factor=2)
    else:
        raise ValueError("Unknown model name.")

    return model.main(**model_param_dict)


def main(model_param_dict, data_param_dict, model_name, hardware):
    output_dict = dict()
    print("-----------------------------------")
    formatted_params = "\n".join(
        f"\t\t{key}: {value}" for key, value in model_param_dict.items()
    )
    nice_log(
        f"\n\t....Running model *** {model_name} *** for....\n"
        f"\t....{model_param_dict['num_bytes']} bytes,\n"
        f"\t....with {data_param_dict['target_bits']} target bits,\n"
        f"\t....with {data_param_dict['corr_intensities']} correlation intensities,\n"
        f"\t....with parameters:\n{formatted_params}",
        color="green",
    )
    print("-----------------------------------")
    results_dir = f"{OUTPUT_FILE_PATH}/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    data_target_file = f"{results_dir}/random_bytes.bin"
    sample_target_file = f"{results_dir}/random_bytes_sample.bin"
    model_param_dict["filename"] = data_target_file

    # iterate over all pairs of corr_intensity and target_bits without repetition
    for target_bits, corr_intensity in itertools.product(
        data_param_dict["target_bits"], data_param_dict["corr_intensities"]
    ):
        model_param_dict["target_bits"] = target_bits
        alpha_scaling_factor = corr_intensity
        beta = 1 - alpha_scaling_factor
        random_bytes, alpha = generate_gbAR_random_bytes(
            alpha_scaling_factor,
            data_param_dict,
            beta,
            model_param_dict["num_bytes"],
        )
        save_random_data(random_bytes, data_target_file, sample_target_file)
        # Running entropy calculations on data
        p_c_random_bytes = calculate_p_c(random_bytes)
        min_entropy_th = arp.ar_min_entropy_limit(beta)
        # Running NIST entropy assessment in parallel with the model
        with ProcessPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(
                run_entropy_assessment, ENTROPY_TEST_BINARY, sample_target_file
            )
            future2 = executor.submit(execute_model, model_name, model_param_dict)
            entropies_dict = future1.result()
            ml_results = future2.result()

        constant_dict = create_constant_dict(
            model_name,
            model_param_dict,
            hardware,
            corr_intensity,
            data_param_dict,
            p_c_random_bytes,
            min_entropy_th,
            entropies_dict,
        )

        for partial_eval in ml_results["eval_results"]:
            run_constant_info = ml_results.copy()
            run_constant_info.pop("eval_results", None)

            eval_result = partial_eval["eval"]
            bytes_processed_eval = partial_eval["bytes_processed_eval"]
            min_entropy_estimated = experimental_min_entropy(
                eval_result["P_ML"], target_bits
            )

            output_dict = {
                **constant_dict,
                **run_constant_info,
                **eval_result,
                "bytes_processed_eval": bytes_processed_eval,
                "min_entropy_estimated": min_entropy_estimated,
            }

            write_results_to_csv(output_dict, results_dir)

        os.remove(data_target_file)
        os.remove(sample_target_file)

        time.sleep(180)

    nice_log(f"Finished running model *** {model_name} ***", color="green")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the main function with custom parameters."
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Model name (default: gpt2)"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Hardware being used (for example: RTX3060Ti or g5.xlarge)",
    )
    parser.add_argument(
        "--corr_intensities",
        type=float,
        nargs="+",
        default=None,
        help="Correlation intensities (default: None)",
    )
    parser.add_argument(
        "--num_bytes",
        type=int,
        default=100000,
        help="Number of bytes (default: 100000)",
    )
    parser.add_argument(
        "--target_bits", type=int, default=None, nargs="+", help="Target bits"
    )
    parser.add_argument(
        "--seqlen", type=int, default=100, help="Max length in bits (default: 100)"
    )
    parser.add_argument("--step", type=int, default=None, help="Step (default: None)")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train ratio (default: 0.8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (default: 10000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs (default: 1)"
    )
    parser.add_argument(
        "--distance_scale_p", type=int, default=1, help="Distance scale p (default: 1)"
    )
    parser.add_argument(
        "--autocorrelation_function",
        type=str,
        default="point-to-point",
        help="Correlation law (default: point-to-point) (exponential, gaussian, point-to-point)",
    )
    parser.add_argument(
        "--signs", type=int, default=None, nargs="+", help="Signs (default: None)"
    )
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
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.hardware is None:
        raise ValueError(
            "Hardware must be specified (for example: RTX3060Ti or g5.xlarge)"
        )

    if args.num_bytes < 10**5:
        raise ValueError("Number of bytes must be at least 10**5 bytes")

    if args.step is None:
        step = args.seqlen
    else:
        step = args.step

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise ValueError("Train ratio must be greater than 0 and lower than 1")

    if args.autocorrelation_function not in [
        "exponential",
        "gaussian",
        "point-to-point",
        "constant",
    ]:
        raise ValueError("Unknown autocorrelation function.")

    if args.corr_intensities is None:
        corr_intensities = np.linspace(
            0, 0.99, num=10
        )  # generates 5 values between 0.3 and 0.5
        # corr_intensities = np.logspace(-2, -0.001, num=20) # this give us more density at the beginning of the interval
    else:
        for corr_intensity in args.corr_intensities:
            if corr_intensity < 0 or corr_intensity > 1:
                raise ValueError("Correlation intensity must be between 0 and 1")
        corr_intensities = args.corr_intensities

    # list of target bits from 1 to 8
    if args.target_bits is None:
        target_bits = [1]
    else:
        for target_bit in args.target_bits:
            if target_bit < 1 or (target_bit > args.seqlen - 1):
                raise ValueError("target_bits must be between 1 and seqlen - 1")
        target_bits = args.target_bits

    exponential_decay_rate = "-"
    gaussian_sigma = "-"
    if args.autocorrelation_function == "exponential":
        exponential_decay_rate = 1 / 10
    elif args.autocorrelation_function == "gaussian":
        gaussian_sigma = args.distance_scale_p / 100

    if args.autocorrelation_function == "constant" and args.signs is not None:
        # check if the number of signs is equal to the distance scale p
        if len(args.signs) != args.distance_scale_p:
            raise ValueError("Number of signs must be equal to the distance scale p")
        for sign in args.signs:
            if sign == 1:
                args.autocorrelation_function += "_+"
            elif sign == -1:
                args.autocorrelation_function += "_-"
            else:
                raise ValueError("Signs must be either 1 or -1")

    for target_bit_n in target_bits:
        check_test_to_classes_ratio(
            args.num_bytes,
            1 - args.train_ratio,
            args.train_ratio,
            target_bit_n,
            args.seqlen,
            step,
        )

    upper_order = num_bytes_order_of_magnitude = int(np.floor(np.log10(1000000)))
    # TODO: generate evaluation checkpoints here if needed
    # evaluation_checkpoints = generate_evaluation_checkpoints(MIN_EVAL_ORDER, upper_order, EVALS_PER_ORDER)
    evaluation_checkpoints = []

    model_param_dict = {
        "generator": f"ar_{args.num_bytes}",
        "num_bytes": int(args.num_bytes),
        "seqlen": args.seqlen,
        "step": step,
        "train_ratio": args.train_ratio,
        "test_ratio": 1 - args.train_ratio,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "evaluation_checkpoints": evaluation_checkpoints,
        "is_autoregressive": args.is_autoregressive,
        "evaluate_all_bits": args.evaluate_all_bits,
    }

    data_param_dict = {
        "num_bytes": args.num_bytes,
        "target_bits": target_bits,
        "corr_intensities": corr_intensities,
        "distance_scale_p": args.distance_scale_p,
        "autocorrelation_function": args.autocorrelation_function,
        "signs": args.signs,
        "exponential_decay_rate": exponential_decay_rate,
        "gaussian_sigma": gaussian_sigma,
    }

    main(
        model_param_dict,
        data_param_dict,
        args.model_name,
        args.hardware,
    )
