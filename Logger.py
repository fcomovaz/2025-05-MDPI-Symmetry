from colorama import Fore, Style  # Set colors for terminal
from time import strftime  # Get current time
import logging  # Set logging

date_format = "%Y-%m-%d %H:%M:%S"


def log_config(embedding: int) -> None:
    str_fmt = f"%Y-%m-%d-Embedding-{embedding}"
    logging.basicConfig(
        level=logging.DEBUG,
        # filename=f"logs/{strftime('%Y-%m-%d-%HH-%MM-%SS')}.log",
        filename=f"logs/{strftime(str_fmt)}.log",
        filemode="w",
        format="[%(levelname)-7s][%(asctime)s] - %(message)s",
        datefmt=date_format,
    )


def log_info(message: str = "") -> None:
    """
    Create a log message with info level custom for the terminal.
    Print it to a log file as well.

    Parameters
    ----------
        message (str): Message to be logged

    Returns
    -------
        None
    """
    print(
        f"[{Fore.BLUE}  INFO {Style.RESET_ALL}]"
        + f"[{strftime(date_format)}] - {message}"
    )
    logging.info(message)


def log_error(message: str = "") -> None:
    """
    Create a log message with info level custom for the terminal.
    Print it to a log file as well.

    Parameters
    ----------
        message (str): Message to be logged

    Returns
    -------
        None
    """
    print(
        f"[{Fore.RED} ERROR {Style.RESET_ALL}]"
        + f"[{strftime(date_format)}] - {message}"
    )
    logging.error(message)


def log_warning(message: str = "") -> None:
    """
    Create a log message with info level custom for the terminal.
    Print it to a log file as well.

    Parameters
    ----------
        message (str): Message to be logged

    Returns
    -------
        None
    """
    print(
        f"[{Fore.YELLOW}WARNING{Style.RESET_ALL}]"
        + f"[{strftime(date_format)}] - {message}"
    )
    logging.warning(message)


def log_success(message: str = "") -> None:
    """
    Create a log message with info level custom for the terminal.
    Print it to a log file as well.

    Parameters
    ----------
        message (str): Message to be logged

    Returns
    -------
        None
    """
    print(
        f"[{Fore.GREEN} DEBUG {Style.RESET_ALL}]"
        + f"[{strftime(date_format)}] - {message}"
    )
    logging.debug(message)


def progress_bar(
    count: int,
    total: int,
    suffix: str = "",
) -> None:
    """
    Display a progress bar.

    Parameters:
    -----------
    count : int
        The current count.
    total : int
        The total count.
    suffix : str, optional
        The suffix to display.

    Returns:
    --------
    None
    """
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = (
        f"{Fore.GREEN}#" * filled_len
        + f"{Fore.RED}_" * (bar_len - filled_len)
        + f"{Style.RESET_ALL}"
    )

    # if count == total-1 print the bar but without the \r
    if count == total - 1:
        print(
            f"[{Fore.GREEN} DEBUG {Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}]"
            + "[%s] %s%s ...%s" % (bar, percents, "%", suffix)
        )
    else:
        print(
            f"[{Fore.BLUE}  INFO {Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}]"
            + "[%s] %s%s ...%s\r" % (bar, percents, "%", suffix),
            end="",
        )
