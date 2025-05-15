import io
import sys
from Logger import *  # logging functions


def log_function_output(function: callable = None) -> None:
    try:
        stream = io.StringIO()  # create StringIO object
        sys.stdout = stream  # and redirect stdout
        function()  # call the function
        sys.stdout = sys.__stdout__  # restore stdout

        # log the output
        for line in stream.getvalue().splitlines():
            log_info(line)
    except Exception as e:
        log_error(f"Error logging {function.__name__} output: {e}")
