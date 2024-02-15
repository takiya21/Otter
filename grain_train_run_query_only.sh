#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}


learning_rate=("1e-4")
warmup_steps_ratio=("0.5")
batch_size=("16")
num_epochs=("5" "10" "20")
num_kf=("0" "1" "2" "3")
# num_kf=("1" "2")
name_respondent=("grain")

dataset="val"

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
            echo instruction_following
            echo "lr="$lr", warmup="$warmup", batch="$batch", epoch="$epoch", kf="$kf", res_name="$res_name

            accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
            pipeline/train/instruction_following.py \
            --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
            --mimicit_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/train_kfold"$kf"_instructions.json" \
            --images_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/train_kfold"$kf"_VI.json" \
            --train_config_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/train_kfold"$kf"_train.json" \
            --val_mimicit_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/val_kfold"$kf"_instructions.json" \
            --val_images_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/val_kfold"$kf"_VI.json" \
            --val_config_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/val_kfold"$kf"_train.json" \
            --external_save_dir="./log" \
            --batch_size=$batch \
            --num_epochs=$epoch \
            --report_to_wandb \
            --wandb_entity=yukiti21 \
            --run_name="one_input_"$res_name"_kf="$kf"_batch"$batch"_lr="$lr"_warmup="$warmup"_epoch="$epoch \
            --wandb_project=Otter_grain \
            --workers=1 \
            --lr_scheduler=cosine \
            --learning_rate=$lr \
            --warmup_steps_ratio=$warmup 

            echo instruction_following_ended

            dataset="val"
            echo val_start
            
            python pipeline/train/test_query_only.py \
            --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
            --mimicit_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/"$dataset"_kfold"$kf"_instructions.json" \
            --images_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/"$dataset"_kfold"$kf"_VI.json" \
            --train_config_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/"$dataset"_kfold"$kf"_train.json" \
            --external_save_dir="./log" \
            --batch_size=$batch \
            --num_epochs=$epoch \
            --wandb_entity=yukiti21 \
            --run_name="one_input_"$res_name"_kf="$kf"_batch"$batch"_lr="$lr"_warmup="$warmup"_epoch="$epoch \
            --workers=1 \
            --learning_rate=$lr \
            --warmup_steps_ratio=$warmup \
            --used_dataset=$dataset \
            --load_ckpt
            echo val_query_only_ended

            dataset="test"
            echo test_start
            
            python pipeline/train/test_query_only.py \
            --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
            --mimicit_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/"$dataset"_kfold"$kf"_instructions.json" \
            --images_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/"$dataset"_kfold"$kf"_VI.json" \
            --train_config_path="/home/taki/Otter/data_for_otter/data/grain_dataset/"$res_name"/"$dataset"_kfold"$kf"_train.json" \
            --external_save_dir="./log" \
            --batch_size=$batch \
            --num_epochs=$epoch \
            --wandb_entity=yukiti21 \
            --run_name="one_input_"$res_name"_kf="$kf"_batch"$batch"_lr="$lr"_warmup="$warmup"_epoch="$epoch \
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
