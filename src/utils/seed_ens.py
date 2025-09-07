import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from src.utils.create_oof import create_oof
import src.utils.telegram as te


def seed_ens(
    tr_df: pd.DataFrame,
    test_df: pd.DataFrame,
    study_name: str,
    trial_num: int,
    ID: int,
    seed_list: list[int],
    full=False,
    iterations=None
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
    full : bool, default False
        Trueの場合、full trainを行う
    iterations : int, default None
        full trainで学習するiteration数
    """
    if full:
        train_mode = "full"
    else:
        train_mode = "single"

    if study_name.startswith("l1_"):
        level = "l1"
    elif study_name.startswith("l2_"):
        level = "l2"
    else:
        level = "base"

    if iterations is None:
        iterations = list(range(len(seed_list)))

    y_true = np.load("../artifacts/y_true.npy")
    all_oof = []
    all_test_preds = []
    score_history = []

    seed_linked = "-".join(str(seed) for seed in seed_list)

    for i, seed in enumerate(seed_list):
        print(f"\n=== SEED {seed} ===")
        oof_list, test_list = create_oof(
            tr_df,
            test_df,
            study_name,
            trial_num_list=[trial_num],
            ID_list=[ID],
            seed=seed,
            full=full,
            iterations_list=[iterations] if full else None,
        )
        all_test_preds.append(test_list[0])

        if not full:
            all_oof.append(oof_list[0])
            oof_array = np.array(all_oof)
            mean_oof = np.mean(oof_array, axis=0)
            tmp_score = roc_auc_score(y_true, mean_oof)
            score_history.append(tmp_score)

            print(f"Tmp Score with {i+1} OOF: {tmp_score:.5f}")

    # test predsの平均化oofの保存
    mean_test = np.mean(np.array(all_test_preds), axis=0)
    test_path = f"../artifacts/preds/{level}/test_{train_mode}_{ID}_seed{seed_linked}.npy"
    np.save(test_path, mean_test)

    if not full:
        mean_oof = np.mean(np.array(all_oof), axis=0)
        oof_path = f"../artifacts/preds/{level}/oof_{train_mode}_{ID}_seed{seed_linked}.npy"
        np.save(oof_path, mean_oof)

        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(score_history)+1),
            score_history,
            marker="o",
            label="AUC")
        plt.xlabel("Number of OOFs added")
        plt.ylabel("AUC")
        plt.title("AUC Seed Ensemble")
        plt.grid(True)
        plt.legend()
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))

        plt.show()

    te.send_telegram_message("SEED ENSEMBLE COMPLETE!")

    return all_oof, all_test_preds