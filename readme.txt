学習実行コード：pipeline/train/instruction_following.py

必要な学習データ（.json）

1枚入力の場合
--mimicit_path= xx_instructions.json
--images_path=xx_VI.json
--train_config_path=xx_train.json

            --mimicit_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/train_kfold"$kf"_instructions.json" \
            --images_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/train_kfold"$kf"_VI.json" \
            --train_config_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/train_kfold"$kf"_train.json" \

複数枚入力の場合
--mimicit_ic_path
--images_ic_path
--train_config_ic_path
        --mimicit_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_no-criteria_instructions.json" \
        --images_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI.json" \
        --train_config_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_pairs25_train.json" \


上記のjsonを作成するコード：/home/taki/Otter/data_for_otter/data/grain_dataset/generate_respondent_json.ipynb（木目,１枚入力用） or /home/taki/Otter/data_for_otter/generate_json.ipynb（目玉焼き、複数枚入力と１枚入力両方）

test用のコード
/home/taki/Otter/pipeline/train/test_query_only.py（1枚入力）
/home/taki/Otter/pipeline/train/test.py（複数枚入力）
	NUM=40をNUM = len(keys)に変えよう
