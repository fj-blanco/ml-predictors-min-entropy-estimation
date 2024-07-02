from datetime import datetime


def nice_log(message, color="yellow", *args):
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    colors = {
        "blue": "\033[0;34m",
        "green": "\033[0;32m",
        "yellow": "\033[0;33m",
        "red": "\033[0;31m",
    }

    if color not in colors:
        raise ValueError(
            f"Invalid color: {color}. Allowed values are: {', '.join(colors.keys())}"
        )

    color_code = colors[color]
    reset_color = "\033[0m"

    log_entry = "{} {}RNG ML: {} ".format(time_str, color_code, reset_color)
    log_entry += str(message) % args

    print(log_entry)


if __name__ == "__main__":
    print("This is the nice_log module.")
