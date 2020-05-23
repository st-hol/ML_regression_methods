def to_num(s):
    """
        util
    """
    try:
        return int(s)
    except ValueError:
        return float(s)