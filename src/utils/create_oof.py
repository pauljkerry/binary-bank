import os
import json
import numpy as np
import pandas as pd
from src.utils.get_trainer import get_trainer
import src.utils.telegram as te

TRAINER_LIST = ["xgb", "lgbm", "cb", "rfr", "rfc", "mlp", "logreg", "ridge"]


def create_oof(
    tr_df: pd.DataFrame,
    test_df: pd.DataFrame,
    study_name: str,
    trial_num_list: list[int],
    ID_list: list[int],
    seed=42,
    full=False,
    iterations_list=None
):
    """
    指定したtrialのparamsを読み込んでOOFを作成し保存

    Parameters
    ----------
    tr_df : pd.DataFrame
        学習用データ
    test_df : pd.DataFrame
        テストデータ
    study_name : str
        保存した study の名前
    trial_num_list : list[int]
        使用するtrial番号のリスト
    ID_list : list[int]
        保存ファイル名に付与する ID
    seed : int, default 42
        seed値
    full : bool, default False
        Trueでfull trainの実施
    iterations_list : list[int]
        full trainで使用するiterations

    Returns
    -------
    all_oof : list[np.ndarray]
        oofを格納したリスト
    all_test_preds : list[np.ndarray]
        testの予測値を格納したリスト
    """
    base_dir = "../artifacts/params"
    study_dir = os.path.join(base_dir, study_name)

    if study_name.startswith("l1_"):
        level = "l1"
    elif study_name.startswith("l2_"):
        level = "l2"
    else:
        level = "base"

    trainer_class = None
    for model in TRAINER_LIST:
        if f"{model}_" in study_name:
            model_type = model
            trainer_class = get_trainer(model_type)
            break
    if trainer_class is None:
        raise ValueError(f"Unsupported study_name: {study_name}")

    all_oof = []
    all_test_preds = []
    preds_path = f"../artifacts/preds/{level}/"

    for i, (trial_num, ID) in enumerate(zip(trial_num_list, ID_list)):
        # --- params をロード ---
        json_path = os.path.join(study_dir, f"trial_{trial_num}.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        params = data["params"]

        # --- 学習 ---
        trainer = trainer_class(
            tr_df,
            test_df,
            params=params,
            seed=seed
        )

        if full:
            if iterations_list is None or len(iterations_list) != len(trial_num_list):
                raise ValueError("When full=True, you must provide iterations_list with the same length as trial_num_list.")

            test_preds = trainer.full_train(iterations_list[i])

            all_test_preds.append(test_preds)
            test_path = os.path.join(
                preds_path, f"test_full_{ID}_seed{seed}.npy")

            np.save(test_path, test_preds)
            continue

        oof, test_preds = trainer.fit()

        all_oof.append(oof)
        all_test_preds.append(test_preds)

        oof_path = os.path.join(
            preds_path, f"oof_single_{ID}_seed{seed}.npy")
        test_path = os.path.join(
            preds_path, f"test_single_{ID}_seed{seed}.npy")

        np.save(oof_path, oof)
        np.save(test_path, test_preds)

    te.send_telegram_message(f"{model.upper()} Training Complete!")

    return all_oof, all_test_preds