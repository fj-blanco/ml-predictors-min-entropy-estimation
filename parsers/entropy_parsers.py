import numpy as np


def parse_iid_and_non_iid_entropy(output):
    non_iid_entropy_value = None
    mcv_bitstring_value = None
    mcv_byte_value = None

    for line in output.splitlines():
        if "min(H_original, 8 X H_bitstring):" in line:
            try:
                non_iid_entropy_value = float(line.split(":")[1].strip())
            except (IndexError, ValueError):
                print(
                    "Unexpected format in entropy output. Unable to parse entropy value."
                )
                non_iid_entropy_value = None

        if "Most Common Value Estimate (bit string)" in line:
            try:
                mcv_bitstring_value = float(line.split("=")[1].split("/")[0].strip())
            except (IndexError, ValueError):
                print(
                    "Unexpected format in entropy output. Unable to parse MCV bitstring value."
                )
                mcv_bitstring_value = None

        if "Most Common Value Estimate =" in line:
            try:
                mcv_byte_value = float(line.split("=")[1].split("/")[0].strip())
            except (IndexError, ValueError):
                print(
                    "Unexpected format in entropy output. Unable to parse MCV literal value."
                )
                mcv_byte_value = None
    return non_iid_entropy_value, mcv_bitstring_value, mcv_byte_value


def parse_predictors_entropies(output):
    output_lines = output.splitlines()
    predictor_names = [
        "Bitstring MultiMCW",
        "Literal MultiMCW",
        "Bitstring Lag",
        "Literal Lag",
        "Bitstring MultiMMC",
        "Literal MultiMMC",
        "Bitstring LZ78Y",
        "Literal LZ78Y",
    ]
    predictors_dict = dict()
    output_dict = dict()
    no_p_local = "Plocal can't affect result"

    def get_local_and_global(line):
        try:
            str_to_find = "Pglobal' = "
            p_global = float(
                line[line.find(str_to_find) + len(str_to_find) :].strip().split(" ")[0]
            )
            if no_p_local in line:
                p_local = "-"
            else:
                str_to_find = "Plocal = "
                p_local = float(
                    line[line.find(str_to_find) + len(str_to_find) :]
                    .strip()
                    .split(" ")[0]
                )
        except (IndexError, ValueError):
            print(
                "Unexpected format in entropy output. Unable to parse predictor global/local entropy value."
            )
            p_local = "-"
            p_global = "-"
        return p_local, p_global

    def get_pred_min_entropy(result_line):
        try:
            pred_min_entropy = float(
                result_line[result_line.find("=") + 1 :].strip().split(" ")[0]
            )
        except (IndexError, ValueError):
            print(
                "Unexpected format in entropy output. Unable to parse predictor result entropy value."
            )
            pred_min_entropy = "-"
        return pred_min_entropy

    for i, line in enumerate(output_lines):
        for pred_name in predictor_names:
            if pred_name in line:
                plocal, pglobal = get_local_and_global(line)
                pglobal_min_entropy = -np.log2(pglobal)
                result_line = output_lines[i + 1]
                pred_min_entropy = get_pred_min_entropy(result_line)
                if "8 bit" in result_line:
                    pglobal_min_entropy = pglobal_min_entropy / 8
                    pred_min_entropy = pred_min_entropy / 8
                predictors_dict[pred_name] = dict(
                    pglobal=pglobal,
                    plocal=plocal,
                    pglobal_min_entropy=pglobal_min_entropy,
                    pred_min_entropy=pred_min_entropy,
                )
    output_dict = {}
    for key, value in predictors_dict.items():
        output_dict[f"{key}_pglobal"] = value["pglobal"]
        output_dict[f"{key}_plocal"] = value["plocal"]
        output_dict[f"{key}_min_entropy"] = value["pred_min_entropy"]

    return output_dict


def parse_entropy_output(output):
    (
        non_iid_entropy_value,
        mcv_bitstring_value,
        mcv_byte_value,
    ) = parse_iid_and_non_iid_entropy(output)

    if mcv_bitstring_value is not None and mcv_byte_value is not None:
        iid_entropy_value = min(mcv_byte_value, 8 * mcv_bitstring_value)
    else:
        iid_entropy_value = None

    predictors_entropies = parse_predictors_entropies(output)
    combined_dict = {
        "entropy_non_iid_nist": non_iid_entropy_value,
        "entropy_iid_nist": iid_entropy_value,
        **predictors_entropies,
    }

    return combined_dict


if __name__ == "__main__":
    print("Parsers module.")
