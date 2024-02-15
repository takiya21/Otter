# 修論付録

修論実験用プログラムの説明書

作成：2024/02/15 滝之弥


## 感性指標構築（語句選別まで）
### データセット解析までの流れ
1. データセットの作成
    - data_for_otter/data/training_data/generate_respondent_json.ipynb
    - データセット
        - xx_instructions.json:質問や回答が入っている
        - xx_train.json:読み込むデータ
        - xx_VI.json:encodeされた画像の情報が入っている
2. Otterの学習と評価
    - 目玉焼きの学習とテスト
        - 1枚入力：run_query_only.shを参照
        - ICL形式：run_VI.json
    - 木目の学習とテスト
        - grain256_train_query_onry.sh
    - 評価ファイル
        - pipline/train/test_query_only.py：logの出力を引数で管理していないので注意。


### ファイルについて
| /5_code/Otter/ | 説明 |
| - | - |
| data_for_otter/data/training_data/h-r | 回答者１ |
| data_for_otter/data/training_data/k-y | 回答社２ |
| data_for_otter/data/training_data/t-h | 回答者３ |
| data_for_otter/data/training_data/a-y | 回答者４ |
| data_for_otter/data/training_data/y-k | 回答者５ |
| data_for_otter/data/training_data/0 | 事前準備データ。ICL形式も含む |