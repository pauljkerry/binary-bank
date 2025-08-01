## 目次

1. [コンペ概要](#1-コンペ概要)
2. [環境](#環境)
3. [ディレクトリ構成](#ディレクトリ構成)
4. [学びと課題](#学びと課題)
5. [実験結果](#実験結果)

## 1. コンペ概要
- **コンペ名**: [Kaggle Playground Series - S5E5](https://www.kaggle.com/competitions/playground-series-s5e5)
- **タスク**: 回帰
- **評価指標**: RMSLE
- **参加タイミング**: コンペ終了後に参加（Late Submission）
- **Privateスコア**: 0.05852
- **順位相当**: 24位 / 4316チーム中（上位 0.55%）

## 2. 環境
### 使用言語・主要ライブラリ

### ▶ MLP（PyTorch）実行環境

| パッケージ名   | バージョン  |
|---------------|------------|
| Python        | 3.11.13    |
| PyTorch       | 2.2.0      |
| torchvision   | 0.17.0     |
| scikit-learn  | 1.6.1      |
| CUDA          | 12.1       |
| numpy         | 1.26.4     |
| pandas        | 2.2.3      |
| optuna        | 4.4.0      |

### ▶ その他の実行環境

| パッケージ名          | バージョン  |
|----------------------|------------|
| Python               | 3.10       |
| XGBoost              | 3.0.2      |
| LightGBM             | 4.6.0      |
| CatBoost             | 1.2.8      |
| scikit-learn         | 1.7.0      |
| CUDA                 | 11.8       |
| cuDF                 | 23.12.01   |
| cuML                 | 23.12.00   |
| CuPy                 | 13.4.1     |
| numpy                | 2.1.3      |
| pandas               | 2.2.3      |
| optuna               | 4.4.0      |
| matplotlib           | 3.10.0     |
| seaborn              | 0.13.2     |


### 環境変数
| 変数名  | 説明       | 例            |
|--------|-------------------|---------------|
| `OPTUNA_STORAGE_URL` | Optunaが使用するPostgreSQLの接続 URL  | `postgresql://user:pass@localhost:5432/db`  |
|`TELEGRAM_CHAT_ID`|Telegram 通知を送る先のチャット ID| `1234567890` |
|`TELEGRAM_TOKEN`|Telegram BotのAPIトークン|`123456789:AAExampleTokenGoesHere`|

## 3. ディレクトリ構成

```text
project/
├── notebooks/                 # Jupyterノート群（EDA, Tuning, 分析など）
├── src/                       # モデル・パイプライン・ユーティリティ群
│   ├── models/                # 各モデルのCV・Optunaコード
│   ├── pipeline/              # 特徴量作成や前処理
│   └── utils/                 # 汎用関数、通知など
├── artifacts/                 # 生成された特徴量・予測結果・前処理済みデータ
├── input/                     # 提供されたデータ
├── output/                    # 提出用CSV
├── .env                       # 環境変数ファイル（※中身はGit管理外）
└── README.md                  # 説明書
```

## 4. 学びと課題

コンペを通して学んだ点や今後の改善点などを以下のMarkdownにまとめました。

 [学びと課題](./summary.md)

## 5. 実験結果
Optunaによるハイパーパラメータ探索の詳細結果は、以下のNotionページでご覧いただけます。

 [Notionリンク](https://www.notion.so/Calorie-portfolio-23dfeb435b01809390a2e5a02625d819?source=copy_link)