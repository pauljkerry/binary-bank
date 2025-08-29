import os
import json
import numpy as np
import 
from sklearn.metrics import roc_auc_score
from src.models.mlp.mlp_cv_trainer import MLPCVTrainer
import src.utils.telegram as te

TRAINER_MAPPING = {
    "mlp": (MLPCVTrainer, "MLP"),
}


def seed_ens(
    tr_df,
    test_df,
    study_name,
    trial_num,
    ID,
    seed_list
):
    """
    指定した trial の params を読み込んで OOF を作成し保存

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
    ID : int
        保存ファイル名に付与するID
    seed_list : list[int]
        seed ensembleを行うseedのリスト
    """
    y_true = tr_df["target"].to_numpy()
    base_dir = "../artifacts/params"
    study_dir = os.path.join(base_dir, study_name)

    if study_name.startswith("l1_"):
        level = "l1"
    elif study_name.startswith("l2_"):
        level = "l2"
    else:
        level = "base"

    trainer_class = None
    for key, value in TRAINER_MAPPING.items():
        if key in study_name:
            trainer_class, model_name_display = TRAINER_MAPPING[key]
            break
    if trainer_class is None:
        raise ValueError(f"Unsupported study_name: {study_name}")

    all_oof = []
    all_test_preds = []
    score_history = []

    for i, seed in enumerate(seed_list):
        # --- params をロード ---
        json_path = os.path.join(study_dir, f"trial_{trial_num}.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        params = data["params"]
        params["seed"] = seed

        # --- 学習 ---
        trainer = trainer_class(
            tr_df,
            test_df,
            params=params
        )

        oof, test_preds = trainer.fit()

        all_oof.append(oof)
        all_test_preds.append(test_preds)

        oof_path = f"../artifacts/preds/{level}/oof_single_{ID}_seed{seed}.npy"
        test_path = f"../artifacts/preds/{level}/test_single_{ID}_seed{seed}.npy"

        np.save(oof_path, oof)
        np.save(test_path, test_preds)

        oof_array = np.array(all_oof)
        test_array = np.array(all_test_preds)

        oof_mean = oof_array.mean(axis=0)
        test_mean = test_array.mean(axis=0)

        tmp_score = roc_auc_score(y_true, oof_mean)
        score_history.append(tmp_score)
        print(f"AUC Round {i+1}: {tmp_score:.5f}")

    seed_str = "-".join(str(s) for s in seed_list)
    oof_mean_path = f"../artifacts/preds/{level}/oof_mean_{ID}_seed{seed_str}.npy"
    test_mean_path = f"../artifacts/preds/{level}/test_mean_{ID}_seed{seed_str}.npy"

    np.save(oof_mean_path, oof_mean)
    np.save(test_mean_path, test_mean)

    te.send_telegram_message(f"{model_name_display} Training Complete!")

    return all_oof, all_test_preds