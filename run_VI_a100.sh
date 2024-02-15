#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}


# learning_rate=("1e-5" "1e-4" "1e-3")
# warmup_steps_ratio=("0.01" "0.1" "0.5")
# batch_size=("16" "32")
# num_epochs=("1" "5" "10" "20")



# accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
# pipeline/train/instruction_following.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --mimicit_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_instructions.json" \
# --images_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI.json" \
# --train_config_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_pairs25_train.json" \
# --val_mimicit_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_instructions.json" \
# --val_images_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI.json" \
# --val_config_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_pairs1_train.json" \
# --external_save_dir="./log" \
# --batch_size=16 \
# --num_epochs=1 \
# --run_name=debug_VI_batch16_pairs5_lr=1e-5_warmup=1e-2_epoch=1 \
# --wandb_project=Otter_egg \
# --workers=1 \
# --lr_scheduler=cosine \
# --learning_rate=1e-5 \
# --warmup_steps_ratio=0.01 \




# for lr in ${learning_rate[@]}
# do
#   for warmup in ${warmup_steps_ratio[@]}
#   do
#     for batch in ${batch_size[@]}
#     do
#       for epoch in ${num_epochs[@]}
#       do
#         echo "lr="$lr", warmup="$warmup", batch="$batch

#         accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
#         pipeline/train/instruction_following.py \
#         --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
#         --mimicit_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_instructions.json" \
#         --images_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI.json" \
#         --train_config_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_pairs25_train.json" \
#         --val_mimicit_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_instructions.json" \
#         --val_images_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI.json" \
#         --val_config_ic_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_pairs1_train.json" \
#         --external_save_dir="./log" \
#         --batch_size=$batch \
#         --num_epochs=$epoch \
#         --report_to_wandb \
#         --wandb_entity=yukiti21 \
#         --run_name="1006_VI_batch"$batch"_pairs5_lr="$lr"_warmup="$warmup"_epoch="$epoch \
#         --wandb_project=Otter_egg \
#         --workers=1 \
#         --lr_scheduler=cosine \
#         --learning_rate=$lr \
#         --warmup_steps_ratio=$warmup 
#       done
#     done
#   done
# done




## query-only learn
#ここから
learning_rate=("1e-3")
warmup_steps_ratio=("0.5")
batch_size=("16" "32")
num_epochs=("10" "20")
dataset="val"

for lr in ${learning_rate[@]}
do
  for warmup in ${warmup_steps_ratio[@]}
  do
    for batch in ${batch_size[@]}
    do
      for epoch in ${num_epochs[@]}
      do
        echo instruction_following
        echo "lr="$lr", warmup="$warmup", batch="$batch

        accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
        pipeline/train/instruction_following.py \
        --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
        --mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_instructions.json" \
        --images_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI.json" \
        --train_config_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_query-only_train.json" \
        --val_mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_instructions.json" \
        --val_images_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI.json" \
        --val_config_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_query-only_train.json" \
        --external_save_dir="./log" \
        --batch_size=$batch \
        --num_epochs=$epoch \
        --report_to_wandb \
        --wandb_entity=yukiti21 \
        --run_name="query-only_VI_batch"$batch"_pairs5_lr="$lr"_warmup="$warmup"_epoch="$epoch \
        --wandb_project=Otter_egg \
        --workers=1 \
        --lr_scheduler=cosine \
        --learning_rate=$lr \
        --warmup_steps_ratio=$warmup 

        echo instruction_following_ended

        echo test_query_only
        python pipeline/train/test_query_only.py \
        --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
        --mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI_instructions.json" \
        --images_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI.json" \
        --train_config_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI_query-only_train.json" \
        --external_save_dir="./log" \
        --batch_size=$batch \
        --num_epochs=$epoch \
        --wandb_entity=yukiti21 \
        --run_name="query-only_VI_batch"$batch"_pairs5_lr="$lr"_warmup="$warmup"_epoch="$epoch \
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


learning_rate=("1e-3")
warmup_steps_ratio=("0.01" "0.1" "0.5")
batch_size=("16" "32")
num_epochs=("50" "100")
dataset="val"

for lr in ${learning_rate[@]}
do
  for warmup in ${warmup_steps_ratio[@]}
  do
    for batch in ${batch_size[@]}
    do
      for epoch in ${num_epochs[@]}
      do
        echo instruction_following
        echo "lr="$lr", warmup="$warmup", batch="$batch

        accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
        pipeline/train/instruction_following.py \
        --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
        --mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_instructions.json" \
        --images_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI.json" \
        --train_config_path="/home/taki/Otter/data_for_otter/data/training_data/0/train_VI_query-only_train.json" \
        --val_mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_instructions.json" \
        --val_images_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI.json" \
        --val_config_path="/home/taki/Otter/data_for_otter/data/training_data/0/val_VI_query-only_train.json" \
        --external_save_dir="./log" \
        --batch_size=$batch \
        --num_epochs=$epoch \
        --report_to_wandb \
        --wandb_entity=yukiti21 \
        --run_name="query-only_VI_batch"$batch"_pairs5_lr="$lr"_warmup="$warmup"_epoch="$epoch \
        --wandb_project=Otter_egg \
        --workers=1 \
        --lr_scheduler=cosine \
        --learning_rate=$lr \
        --warmup_steps_ratio=$warmup 

        echo instruction_following_ended

        echo test_query_only
        python pipeline/train/test_query_only.py \
        --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
        --mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI_instructions.json" \
        --images_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI.json" \
        --train_config_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI_query-only_train.json" \
        --external_save_dir="./log" \
        --batch_size=$batch \
        --num_epochs=$epoch \
        --wandb_entity=yukiti21 \
        --run_name="query-only_VI_batch"$batch"_pairs5_lr="$lr"_warmup="$warmup"_epoch="$epoch \
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
