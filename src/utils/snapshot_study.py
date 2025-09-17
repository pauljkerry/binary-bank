import os
import optuna
import optuna.visualization as vis
from pathlib import Path

from src.utils.telegram import send_message, send_document


def snapshot_study(
    study: optuna.Study,
    study_name: str,
    trial_num: int,
    out_root: str = "../artifacts/optuna",
    send_telegram: bool = True
) -> list[str]:
    """
    Studyの現状を CSV + グラフ に保存し、作成ファイルのパス一覧を返す。
    """
    base = Path(out_root) / f"{study_name}/trl{trial_num}"
    os.makedirs(base, exist_ok=True)

    created: list[str] = []

    # 2) 可視化
    figs = [
        ("opt_history",        vis.plot_optimization_history(study)),
        ("param_importances",  vis.plot_param_importances(study)),
        ("parallel_coord",     vis.plot_parallel_coordinate(study)),
    ]

    try:
        for name, fig in figs:
            p = base / f"{name}.html"
            fig.write_html(p, include_plotlyjs="cdn")
            created.append(str(p))

    except Exception as e:
        print(f"[snapshot] html export failed ({e}).")

    # 3) Telegram（任意）
    if send_telegram:
        msg = f"[{study_name}] snapshot at trial {trial_num}"
        send_message(msg)

        # 画像があれば画像、なければHTMLを全部送る
        to_send = [p for p in created if p.endswith(".html")]

        for path in to_send:
            send_document(path)

    return created