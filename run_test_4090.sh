#/bin/bash
export PYTHONPATH=.
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}


# learning_rate=("1e-5" "1e-4" "1e-3")
# warmup_steps_ratio=("0.01" "0.1" "0.5")
# batch_size=("16" "32")
# num_epochs=("1" "5")


learning_rate=("1e-4")
warmup_steps_ratio=("0.01")
batch_size=("32")
num_epochs=("5")
dataset="test"

# learning_rate=("1e-4")
# warmup_steps_ratio=("0.1")
# batch_size=("16")
# num_epochs=("10")
# dataset="test"

for lr in ${learning_rate[@]}
do
  for warmup in ${warmup_steps_ratio[@]}
  do
    for batch in ${batch_size[@]}
    do
      for epoch in ${num_epochs[@]}
      do

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
        --run_name="ICL_with-criteria_VI_batch"$batch"_pairs5_lr="$lr"_warmup="$warmup"_epoch="$epoch \
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


# learning_rate=("1e-4")
# warmup_steps_ratio=("0.1")
# batch_size=("32")
# num_epochs=("5")
# dataset="test"

# for lr in ${learning_rate[@]}
# do
#   for warmup in ${warmup_steps_ratio[@]}
#   do
#     for batch in ${batch_size[@]}
#     do
#       for epoch in ${num_epochs[@]}
#       do

#         echo test_query_only
#         python pipeline/train/test_query_only.py \
#         --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
#         --mimicit_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI_instructions.json" \
#         --images_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI.json" \
#         --train_config_path="/home/taki/Otter/data_for_otter/data/training_data/0/"$dataset"_VI_query-only_train.json" \
#         --external_save_dir="./log" \
#         --batch_size=$batch \
#         --num_epochs=$epoch \
#         --wandb_entity=yukiti21 \
#         --run_name="ICL_with-criteria_VI_batch"$batch"_pairs5_lr="$lr"_warmup="$warmup"_epoch="$epoch \
#         --workers=1 \
#         --learning_rate=$lr \
#         --warmup_steps_ratio=$warmup \
#         --used_dataset=$dataset \
#         --load_ckpt
#         echo test_query_only_ended
#       done
#     done
#   done
# done

## for zero-shot test

# python pipeline/train/test.py \
# --pretrained_model_name_or_path="./weights/OTTER-Image-MPT7B" \
# --external_save_dir="./log" \
# --run_name=zeroshot-testdata_force-answer \
# --workers=1 \
# --used_dataset=test