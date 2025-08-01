import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from IPython.display import display


def check_missing_info(df):
    """
    欠損値の情報を確認する関数。

    Parameters
    ----------
    df : pd.DataFrame
        欠損値を確認するデータ。
    """
    missing_counts = df.isna().sum()

    missing_more_than_0 = missing_counts[missing_counts > 0]
    if missing_more_than_0.empty:
        print("No missing")
        return

    missing_ratio = missing_more_than_0 / len(df)

    missing_ratio_percent = missing_ratio.apply(
        lambda x: f"{x*100:.2f}%"
    )

    missing_dtypes = df.dtypes[missing_more_than_0.index]

    # 欠損数と欠損割合をまとめたDFを作成
    missing_df = pd.DataFrame({
        "missing_count": missing_more_than_0,
        "missing_ratio": missing_ratio_percent,
        "dtype": missing_dtypes
    }).sort_values("missing_count")

    cat_mask = missing_df["dtype"].astype(str).isin(["object", "category"])
    missing_cat = missing_df[cat_mask].sort_values(
        "missing_count", ascending=False)
    missing_num = missing_df[~cat_mask].sort_values(
        "missing_count", ascending=False)

    if missing_cat.empty:
        print("No missing values in cat features")
    else:
        print(missing_cat)
    if missing_num.empty:
        print("No missing values in num features")
    else:
        print(missing_num)


def plot_category_freq(df, cols=None):
    """
    カテゴリ変数の頻度を棒グラフと表形式で表示する関数。

    Parameters
    ----------
    df : pd.DataFrame
        カテゴリ変数の分布を確認するデータ。
    cols : list of str or None, default None
        カテゴリ変数の列名。Noneの場合はすべてのカテゴリ変数を対象とする。
    """
    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns

    n_cols = 2
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    summary_tables = {}

    for i, col in enumerate(cols):
        cat_levels = df[col].value_counts(dropna=False).index

        # グラフ描画
        ax = axes[i]
        sns.countplot(
            data=df,
            x=col,
            palette="Set3",
            hue=col,
            hue_order=cat_levels,
            order=cat_levels,
            ax=ax
        )
        ax.set_ylabel("Frequency")
        ax.set_title(f"Category Frequency of {col}")
        ax.tick_params(axis='x', rotation=45)

        # 表形式でも出力
        value_counts = df[col].value_counts(dropna=False).to_numpy()
        value_ratio = (value_counts / value_counts.sum()) * 100
        summary_df = pd.DataFrame({
            "Count": value_counts,
            "Ratio (%)": [f"{r:.2f}%" for r in value_ratio],
        }, index=cat_levels)

        summary_tables[col] = summary_df.T

    # 余ったサブプロットを非表示に
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    for col, summary_df_T in summary_tables.items():
        display(summary_df_T)


def plot_numerical_distribution(df, cols=None, bins=30):
    """
    数値変数の分布を表示する関数。

    Parameters
    ----------
    df : pd.DataFrame
        表示するデータ。
    cols : list or None, default None
        数値変数の列名。Noneのときはすべての数値変数の列名。
    bins : int, default 30
        bin分割の個数
    """
    # colの指定
    if cols is None:
        cols = sorted(df.select_dtypes(include=np.number).columns)

    # グリッドの行列数の設定
    n_cols = 2
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    axes = axes.flatten()  # 1次元配列に変換

    stats_dict = {}

    for i, col in enumerate(cols):
        series = df[col].dropna()
        ax = axes[i]
        sns.histplot(
            df[col].dropna(),
            kde=True,
            bins=bins,
            color="skyblue",
            ax=ax
        )
        ax.set_title(f"Distribution of {col}")
        ax.set_ylabel("Frequency")

        # 統計量を取得
        stats = {
            "count": series.count(),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "1%": series.quantile(0.01),
            "5%": series.quantile(0.05),
            "10%": series.quantile(0.10),
            "25%": series.quantile(0.25),
            "50%": series.quantile(0.50),
            "75%": series.quantile(0.75),
            "90%": series.quantile(0.90),
            "95%": series.quantile(0.95),
            "99%": series.quantile(0.99),
            "max": series.max(),
        }
        stats_dict[col] = stats

    # 余ったサブプロットを非表示に
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    cols_order = [
        "count", "mean", "std",
        "min", "1%", "5%", "10%", "25%", "50%",
        "75%", "90%", "95%", "99%", "max"
    ]
    summary_df = pd.DataFrame(stats_dict).T
    summary_df = summary_df[cols_order]
    display(summary_df)


def plot_kde_and_boxplot(df, num_cols=None, cat_cols=None):
    """
    カテゴリ変数ごとの数値変数に対する分布を表示する関数。

    Parameters
    ----------
    df : pd.DataFrame
        表示するdf。
    num_cols : list or None, default NOne
        数値変数の列名のリスト。Noneの場合はdf内のすべての数値変数。
    cat_cols : list or None, default None
        カテゴリ変数の列名のリスト。Noneの場合はdf内のすべてのカテゴリ変数。
    """
    if cat_cols is None:
        cat_cols = sorted(df.select_dtypes(include=["object", "category"]).columns)
    if num_cols is None:
        num_cols = sorted(df.select_dtypes(include=np.number).columns)

    n_cols = 2
    n_rows = len(cat_cols) * len(num_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)  # 2次元に整形

    row = 0
    for cat_col in cat_cols:
        for num_col in num_cols:
            cat_levels = df[cat_col].value_counts(dropna=False).index

            # 左：カテゴリの頻度
            ax1 = axes[row, 0]
            sns.kdeplot(
                data=df,
                x=num_col,
                hue=cat_col,
                hue_order=cat_levels,
                ax=ax1,
                common_norm=False,
                fill=False,
                linewidth=2,
            )
            ax1.set_title(f"Distribution of {num_col} by {cat_col}", fontsize=14)
            ax1.set_xlabel(num_col)
            ax1.set_ylabel("Density")
            ax1.tick_params(axis="x", rotation=45)

            # 右：箱ひげ図
            ax2 = axes[row, 1]
            sns.boxplot(
                data=df,
                x=cat_col,
                y=num_col,
                order=cat_levels,
                ax=ax2,
                palette="Set3",
                hue=cat_col,
                hue_order=cat_levels,
                legend=False,
                showfliers=False
            )
            ax2.set_title(f"Distribution of {num_col} by {cat_col}", fontsize=14)
            ax2.set_xlabel("")
            ax2.set_ylabel(num_col)
            ax2.tick_params(axis="x", rotation=45)

            row += 1

    plt.tight_layout()
    plt.show()


def show_corr_heatmap(df, cols=None):
    """
    数値変数間の相関係数を表示する関数

    Parameters
    ----------
    df : pd.DataFrame
        相関係数を表示するデータ
    cols : list or None, default None
        数値変数の列名。Noneの場合はすべての数値変数の列名。
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns

    num_df = df[cols]
    corr_df = num_df.corr()
    plt.figure(figsize=(5, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation heatmap')