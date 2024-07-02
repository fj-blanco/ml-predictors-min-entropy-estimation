import subprocess
import numpy as np
from bitarray import bitarray


def generate_openssl_random_bytes(num_bytes):
    max_bytes_per_call = 60000000
    parts = []

    while num_bytes > 0:
        bytes_to_generate = min(num_bytes, max_bytes_per_call)
        command = ["openssl", "rand", str(bytes_to_generate)]
        proc = subprocess.run(command, capture_output=True, check=True)
        parts.append(proc.stdout)
        num_bytes -= bytes_to_generate

    return b"".join(parts)


def ossl_rand_mn_rvs(P_abs, N_bits):
    """
    This function takes the normalized probabilities P_abs and returns N samples from the multinomial distribution.
    """
    num_bytes = N_bits * 4
    random_bytes = generate_openssl_random_bytes(num_bytes)

    for i in range(0, num_bytes, 4):
        random_number = (
            int.from_bytes(random_bytes[i : i + 4], byteorder="big") / 0xFFFFFFFF
        )
        cumulative_prob = 0.0
        sample = np.zeros(len(P_abs), dtype=int)
        for i, prob in enumerate(P_abs):
            cumulative_prob += prob
            if random_number <= cumulative_prob:
                sample[i] = 1
                break
        yield sample


def test_ossl_rand_mn_rvs():
    P_abs = [0.2, 0.4, 0.1, 0.3, 0.0]  # Example input vector with five components
    num_runs = 1000000
    counts = [0] * len(P_abs)

    # Generate num_runs samples at once
    samples = ossl_rand_mn_rvs(P_abs, num_runs)

    # Count the occurrences of each outcome
    for sample in samples:
        outcome = np.argmax(sample)
        counts[outcome] += 1

    # Calculate the observed probabilities
    observed_probabilities = [count / num_runs for count in counts]

    # Check that the observed probabilities are close to the expected probabilities
    tolerance = 0.02
    for expected, observed in zip(P_abs, observed_probabilities):
        assert (
            abs(expected - observed) < tolerance
        ), f"Expected {expected}, but observed {observed}"


def point_to_point_alpha(p, scaling_factor):
    """
    Generate a normalized point-to-point sequence of length p
    p: length of the sequence
    scaling_factor: norm of the sequence
    """
    alpha = np.zeros(p)
    alpha[p - 1] = 1

    return scaling_factor * alpha


def constant_alpha(p, scaling_factor, signs=None):
    """
    Generate a normalized constant sequence of length p
    p: length of the sequence
    scaling_factor: norm of the sequence
    """
    if signs is None:
        alpha = np.ones(p) / p
    else:
        alpha = np.array([x / len(signs) for x in signs])

    return scaling_factor * alpha


def exponentially_decreasing_alpha(p, scaling_factor, decay_rate=1):
    """
    Generate a normalized exponentially decreasing sequence of length p
    p: length of the sequence
    decay_rate: rate of exponential decay
    """
    alpha = np.array([np.exp(-decay_rate * i) for i in range(p)])

    alpha /= np.sum(alpha)

    return scaling_factor * alpha


def gaussian_alpha(mu, scaling_factor, sigma, threshold=0):
    """
    Generate a normalized Gaussian sequence of length p centered at mu
    p: length of the sequence
    mu: the mean value (center of the Gaussian)
    sigma: the standard deviation of the Gaussian
    threshold: minimum value to include in the sequence
    """
    p = mu + 10 * sigma
    # Create an array of Gaussian values
    x = np.arange(p)
    alpha = np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    # Apply the threshold to the right tail
    alpha[mu:][alpha[mu:] < threshold] = 0

    # Remove trailing zeros from the right tail
    alpha = np.trim_zeros(alpha, trim="b")

    # remove the first element (p=0)
    alpha = alpha[1:]

    # Normalize the sequence so that its sum is 1
    alpha /= np.sum(alpha)

    if len(alpha) != mu:
        raise ValueError("The length of the sequence does not match the mean value.")

    return scaling_factor * alpha


def ar_min_entropy_limit(beta):
    entropy = -np.log2(1 - beta / 2)
    return entropy


def gbAR(alpha, beta, N_bytes, byteshift=10**4):
    """
    Generalized Binary AR Process
    alpha: vector of alpha parameters
    beta: beta parameter
    et: error process
    p: order of AR process
    N: number of timesteps
    """
    p = len(alpha)
    bitshift = 8 * byteshift
    N_bits = N_bytes * 8 + bitshift

    P_abs = np.abs(np.concatenate([alpha, np.array([beta])]))
    P_abs /= np.sum(P_abs)

    Xt_window = [False] * p
    final_bits = bitarray()

    random_noise_bytes = generate_openssl_random_bytes(N_bytes + byteshift)
    et = np.unpackbits(np.frombuffer(random_noise_bytes, dtype=np.uint8))

    for t, sample in enumerate(ossl_rand_mn_rvs(P_abs, N_bits)):
        if t < p:
            continue
        at_plus = sample[:-1] * (alpha >= 0)
        at_minus = sample[:-1] * (alpha < 0)
        bt = sample[-1]

        Xt_bit = (
            np.sum(at_plus * Xt_window[::-1])
            + np.sum(at_minus * (1 - np.array(Xt_window[::-1])))
            + bt * et[t]
        ) % 2

        final_bits.append(Xt_bit)
        Xt_window.pop(0)
        Xt_window.append(Xt_bit)

    return final_bits[-(N_bytes * 8) :].tobytes()


if __name__ == "__main__":
    print("This is a functions module.")
