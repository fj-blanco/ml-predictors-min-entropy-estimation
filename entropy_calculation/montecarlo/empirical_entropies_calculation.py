import os
import sys
import csv
import itertools
import numpy as np

from multiprocessing import Pool

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(module_path)
import autoregressive_process.autoregressive_process as arp


def count_byte_frequencies(random_bytes):
    freq_dict = {}
    total_bytes = len(random_bytes)

    for byte in random_bytes:
        if byte in freq_dict:
            freq_dict[byte] += 1
        else:
            freq_dict[byte] = 1

    # Normalize frequencies
    for key in freq_dict:
        freq_dict[key] /= total_bytes

    return freq_dict


def count_bit_frequencies(random_bytes, p):
    freq_dict = {}
    bit_sequence = "".join([format(byte, "08b") for byte in random_bytes])
    total_bits = len(bit_sequence)
    step_size = p

    for i in range(0, total_bits - step_size + 1, step_size):
        bit_chunk = bit_sequence[i : i + step_size]
        int_val = int(bit_chunk, 2)

        if int_val in freq_dict:
            freq_dict[int_val] += 1
        else:
            freq_dict[int_val] = 1

    # Normalize frequencies
    for key in freq_dict:
        freq_dict[key] /= total_bits / step_size

    return freq_dict


def conditional_prob_max(alpha, beta, x):
    # chech x is greater or equal in length than alpha
    if len(x) < len(alpha):
        raise ValueError("x must be greater or equal in length than alpha")
    p = len(alpha)
    # compute scalar product between alpha and the last p values of x
    scalar_product = np.dot(alpha, x[::-1][:p])
    conditional_prob = scalar_product + beta / 2
    return max(conditional_prob, 1 - conditional_prob)


def compute_average_conditional_min_entropy(alpha, beta, freq_dict):
    result_sum = 0.0

    # Iterate over all possible byte values (0 to 255)
    for byte_val in range(256):
        x = [
            int(bit) for bit in format(byte_val, "08b")
        ]  # Convert to binary representation
        conditional_prob = conditional_prob_max(alpha, beta, x)

        freq = freq_dict.get(byte_val, 0)  # Get frequency, 0 if not in dictionary
        result_sum += freq * conditional_prob

    return -np.log2(result_sum)


def conditional_prob_max_n_bits(alpha, beta, x, n_cond):
    p = len(alpha)
    if len(x) != p:
        raise ValueError("x must be equal in length to alpha")

    x = x[::-1]

    combinations = list(itertools.product([0, 1], repeat=(1 + n_cond)))

    probabilities = []

    for combination in combinations:
        joint_prob = 1.0  # Initialize joint probability

        for i in range(1 + n_cond):  # Use 0-based indexing
            if i < p:
                x_vector = x[-p + i :] + list(combination[:i])
            else:
                x_vector = list(combination[i - p : i])

            scalar_product_pos = np.dot(np.abs(alpha) * (alpha > 0), x_vector[::-1][:p])
            scalar_product_neg = np.dot(
                np.abs(alpha) * (alpha < 0), np.logical_not(x_vector[::-1][:p])
            )

            conditional_prob_1 = scalar_product_pos + scalar_product_neg + beta / 2

            bit = combination[i]

            conditional_prob = (
                conditional_prob_1 if bit == 1 else 1 - conditional_prob_1
            )

            joint_prob *= conditional_prob  # Update joint probability

        probabilities.append(joint_prob)

    return max(probabilities)


def compute_average_conditional_min_entropy_n_bits(alpha, beta, freq_dict, n_cond):
    p = len(alpha)
    result_sum = 0.0

    for byte_val in range(2**p):
        x = [
            int(bit) for bit in format(byte_val, "08b")
        ]  # Convert to binary representation
        if len(x) < p:
            x = [0] * (p - len(x)) + x
        conditional_prob = conditional_prob_max_n_bits(alpha, beta, x[-p:], n_cond)

        freq = freq_dict.get(byte_val, 0)
        result_sum += freq * conditional_prob

    return -np.log2(result_sum) / (1 + n_cond)


def aggregate_frequencies(g, alpha, alpha_scaling_factor, num_bytes, distance_scale_p):
    aggregate_freq_dict = {}

    for _ in range(g):
        random_bytes = arp.gbAR(alpha, 1 - alpha_scaling_factor, num_bytes)
        freq_dict = count_bit_frequencies(random_bytes, distance_scale_p)

        for bit_chunk, freq in freq_dict.items():
            if bit_chunk in aggregate_freq_dict:
                aggregate_freq_dict[bit_chunk] += freq
            else:
                aggregate_freq_dict[bit_chunk] = freq

    # Normalize to get average frequencies
    for bit_chunk in aggregate_freq_dict:
        aggregate_freq_dict[bit_chunk] /= g

    return aggregate_freq_dict


def generate_and_count(
    run_id, alpha, alpha_scaling_factor, num_bytes, distance_scale_p
):
    random_bytes = arp.gbAR(alpha, 1 - alpha_scaling_factor, num_bytes)
    tmp_freq_dict = count_bit_frequencies(random_bytes, distance_scale_p)
    return tmp_freq_dict


def compute_min_entropy_n_bits(alpha, beta, num_bytes, n_cond):
    alpha_scaling_factor = 1 - beta
    distance_scale_p = len(alpha)
    n_target_bits = 1 + n_cond
    freq_dict = get_freq_dict(alpha, alpha_scaling_factor, num_bytes, n_target_bits, 10)
    # get de maximum value of the dictionary
    max_value = max(freq_dict.values())
    entropy = -(1 / n_target_bits) * np.log2(max_value)

    return entropy


def get_freq_dict(
    alpha, alpha_scaling_factor, num_bytes, distance_scale_p, gen_runs=10
):
    with Pool() as pool:
        # Create a list of arguments for each run
        args = [
            (i, alpha, alpha_scaling_factor, num_bytes, distance_scale_p)
            for i in range(gen_runs)
        ]

        # Run the generation in parallel
        results = pool.starmap(generate_and_count, args)

    # Aggregate the results
    freq_dict = {}
    for tmp_freq_dict in results:
        for bit_chunk, freq in tmp_freq_dict.items():
            freq_dict[bit_chunk] = freq_dict.get(bit_chunk, 0) + freq

    # Normalize to get average frequencies
    for bit_chunk in freq_dict:
        freq_dict[bit_chunk] /= gen_runs

    return freq_dict


def generate_average_conditional_min_entropy_n_bits_csv(
    correlation_function,
    distance_scale_p,
    num_bytes,
    n_cond_max,
    target_dir,
    decay_rate=None,
    sigma=None,
    threshold=None,
    signs=None,
):
    master_data_dict = {}
    gen_runs = 10

    MIN_ENTROPY = True
    AVG_MIN_ENTROPY = True

    corr_intensities = np.linspace(0, 0.99, num=10)
    corr_intensities = [0.5]
    for alpha_scaling_factor in corr_intensities:
        beta = 1 - alpha_scaling_factor

        function_name = correlation_function.__name__

        if function_name == "exponentially_decreasing_alpha":
            alpha = correlation_function(
                distance_scale_p, alpha_scaling_factor, decay_rate=decay_rate
            )
        elif function_name == "constant_alpha":
            if signs is None:
                alpha = correlation_function(distance_scale_p, alpha_scaling_factor)
            else:
                alpha = correlation_function(
                    distance_scale_p, alpha_scaling_factor, signs=signs
                )
        elif function_name == "gaussian_alpha":
            alpha = correlation_function(
                distance_scale_p, alpha_scaling_factor, sigma, threshold
            )
            # remove first value of alpha
            alpha = alpha[1:]
        elif function_name == "point_to_point_alpha":
            alpha = correlation_function(distance_scale_p, alpha_scaling_factor)
        else:
            raise ValueError("Invalid correlation function name")

        # Here we get the frequencies dict of bit sequences of length distance_scale_p
        freq_dict = get_freq_dict(
            alpha, alpha_scaling_factor, num_bytes, distance_scale_p, gen_runs
        )

        min_entropy_limit = arp.ar_min_entropy_limit(beta)
        key = (alpha_scaling_factor, min_entropy_limit)

        for n_cond in range(1 + n_cond_max):
            if AVG_MIN_ENTROPY:
                average_conditional_min_entropy = (
                    compute_average_conditional_min_entropy_n_bits(
                        alpha, beta, freq_dict, n_cond
                    )
                )
                # For average conditional min entropy
                avg_key = (key[0], key[1], "avg")
                master_data_dict.setdefault(avg_key, {})
                master_data_dict[avg_key][
                    f"Min-Entropy_n{n_cond}"
                ] = average_conditional_min_entropy
            if MIN_ENTROPY:
                min_entropy = compute_min_entropy_n_bits(alpha, beta, num_bytes, n_cond)
                # For min entropy
                joint_key = (key[0], key[1], "joint")
                master_data_dict.setdefault(joint_key, {})
                master_data_dict[joint_key][f"Min-Entropy_n{n_cond}"] = min_entropy

    if function_name == "constant_alpha" and signs is not None:
        signs_string = ""
        for sign in signs:
            if sign == 1:
                signs_string += "+"
            elif sign == -1:
                signs_string += "-"
            else:
                raise ValueError("Invalid sign value")
        function_name += f"_{signs_string}"

    gbarp_id = f"{function_name}_{n_cond_max}_p_{distance_scale_p}"
    target_file = (
        f"{target_dir}/avg_min_entropy_several_conditional_target_{gbarp_id}.csv"
    )

    with open(target_file, "w", newline="") as csvfile:
        fieldnames = [
            "Alpha Scaling Factor",
            "Min-Entropy Limit",
            "Entropy",
            "gbarp_id",
            "num_bytes",
            "correlation_function",
            "distance_scale_p",
            "n_cond_max",
            "decay_rate",
            "sigma",
            "threshold",
        ] + [f"Min-Entropy_n{i}" for i in range(1 + n_cond_max)]
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        for (
            alpha_scaling_factor,
            min_entropy_limit,
            entropy_type,
        ), values in master_data_dict.items():
            row = {
                "Alpha Scaling Factor": alpha_scaling_factor,
                "Min-Entropy Limit": min_entropy_limit,
                "Entropy": entropy_type,
                "gbarp_id": gbarp_id,
                "num_bytes": num_bytes,
                "correlation_function": correlation_function,
                "distance_scale_p": distance_scale_p,
                "n_cond_max": n_cond_max,
                "decay_rate": decay_rate if decay_rate is not None else "-",
                "sigma": sigma if sigma is not None else "-",
                "threshold": threshold if threshold is not None else "-",
            }
            row.update(values)
            csvwriter.writerow(row)


if __name__ == "__main__":
    num_bytes = 100000
    n_cond_max = 16
    distance_scale_p = 2

    # TODO: change the alpha generating function, you can use the following functions:
    # {arp.exponentially_decreasing_alpha, arp.point_to_point_alpha, arp.gaussian_alpha, arp.constant_alpha, arp.constant_alpha}
    alpha_generating_function = arp.constant_alpha

    # TODO: change the signs to generate the constant alpha sequence
    # for example signs = [1, -1]
    signs = None

    generate_entropy_dataset = True

    target_dir = os.path.join(
        CURRENT_DIR, "..", "..", "data_analysis", "results", "montecarlo"
    )

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # DEPRECATED
    if generate_entropy_dataset:
        if alpha_generating_function.__name__ == "exponentially_decreasing_alpha":
            decay_rate = 1 / 10
            generate_average_conditional_min_entropy_n_bits_csv(
                alpha_generating_function,
                distance_scale_p,
                num_bytes,
                n_cond_max,
                target_dir,
                decay_rate=decay_rate,
            )
        elif alpha_generating_function.__name__ == "gaussian_alpha":
            sigma = distance_scale_p / 100  # example value for sigma
            threshold = 0.001
            generate_average_conditional_min_entropy_n_bits_csv(
                alpha_generating_function,
                distance_scale_p,
                num_bytes,
                n_cond_max,
                target_dir,
                sigma=sigma,
                threshold=threshold,
            )
        elif alpha_generating_function.__name__ == "constant_alpha":
            alpha = 0.5
            generate_average_conditional_min_entropy_n_bits_csv(
                alpha_generating_function,
                distance_scale_p,
                num_bytes,
                n_cond_max,
                target_dir,
                signs=signs,
            )
        else:
            generate_average_conditional_min_entropy_n_bits_csv(
                alpha_generating_function,
                distance_scale_p,
                num_bytes,
                n_cond_max,
                target_dir,
            )
