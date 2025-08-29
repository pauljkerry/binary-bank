import optuna
import optuna.visualization as vis
import os
import json


class OptunaVisualizer:
    """
    Optunaの探索結果を表示するクラス

    Attributes
    ----------
    study_name : str
        StudyName
    storage_path : str
        ストレージの保存先
    """

    def __init__(self, study_name, storage_path):
        self.study_name = study_name
        self.storage_path = storage_path
        self.study = optuna.load_study(
            study_name=study_name,
            storage=storage_path
        )

    def visualize_optimization(self):
        """
        探索結果を可視化する関数。

        Notes
        -----
        - パラメータ重要度
        - 最適化履歴
        - パラメータの相互関係
        """
        # パラメータ重要度
        fig1 = vis.plot_param_importances(self.study)
        fig1.show()

        # 最適化履歴
        fig2 = vis.plot_optimization_history(self.study)
        fig2.show()

        # パラメータの相互依存関係
        fig3 = vis.plot_parallel_coordinate(self.study)
        fig3.show()

    def save_top_params(self, top_k=3):
        """
        Optuna study の trial 結果を表示・保存

        Parameters
        ----------
        top_k : int, default 3
            保存するtrialの上位数
        """
        # trialの取得とソート
        trials = [t for t in self.study.trials if t.value is not None]
        reverse = self.study.direction == optuna.study.StudyDirection.MAXIMIZE
        sorted_trials = sorted(trials, key=lambda t: t.value, reverse=reverse)

        # study ごとのディレクトリ作成
        base_dir = "../artifacts/params"
        study_dir = os.path.join(base_dir, self.study.study_name)
        os.makedirs(study_dir, exist_ok=True)

        print(f"=== Top {top_k} Trials ===")

        for t in sorted_trials[:top_k]:
            print(f"=== Trial {t.number} ===")
            print(f"CV Score       : {t.value:.5f}")

            # params の値を文字列に変換（文字列はクォート付き）
            """safe_params = {k: repr(v) if isinstance(v, str) else v
                           for k, v in t.params.items()}"""

            data = {
                "trial_number": t.number,
                "value": t.value,
                "params": t.params
            }
            filename = f"trial_{t.number}.json"
            path = os.path.join(study_dir, filename)
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Saved params: {path}")