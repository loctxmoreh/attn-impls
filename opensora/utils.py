def get_error_message(exception):
    exception_str = str(exception).strip()
    if len(exception_str):
        return exception_str.split("\n")[-1].strip()
    else:
        return "Empty error message."
