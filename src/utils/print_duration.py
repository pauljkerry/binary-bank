def print_duration(start, end, label="Training time"):
    """
    経過時間を出力する関数。

    Parameters
    ----------
    start : float
        開始時間
    end : float
        終了時間
    label : str, default "Training time"
        出力時に先頭に表示するラベル
    """
    duration = end - start
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"{label}: "
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{int(seconds):02d}"
    )