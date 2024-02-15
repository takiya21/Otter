#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}


learning_rate=("1e-4")
warmup_steps_ratio=("0.5")
batch_size=("16")
num_epochs=("50")
num_kf=("0" "1" "2" "3" "4")
name_respondent=("k-y")

dataset="test"

for lr in ${learning_rate[@]}
do
  for warmup in ${warmup_steps_ratio[@]}
  do
    for batch in ${batch_size[@]}
    do
      for epoch in ${num_epochs[@]}
      do
        for kf in ${num_kf[@]}
        do
          for res_name in ${name_respondent[@]}
          do 


            echo test_query_only
            python pipeline/train/test_query_only.py \
            --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
            --mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/"$res_name"/test_instructions.json" \
            --images_path="/home/taki/Otter/data_for_otter/data/training_data/"$res_name"/test_VI.json" \
            --train_config_path="/home/taki/Otter/data_for_otter/data/training_data/"$res_name"/test_train.json" \
            --external_save_dir="./log" \
            --batch_size=$batch \
            --num_epochs=$epoch \
            --wandb_entity=yukiti21 \
            --run_name="one_input_"$res_name"kf="$kf"_batch"$batch"_lr="$lr"_warmup="$warmup"_epoch="$epoch \
            --workers=1 \
            --learning_rate=$lr \
            --warmup_steps_ratio=$warmup \
            --used_dataset=$dataset \
            --load_ckpt
            echo test_query_only_ended
          done
        done
      done
    done
  done
done
