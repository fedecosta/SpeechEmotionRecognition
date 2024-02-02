#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
python scripts/train.py \
	--train_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--validation_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--augmentation_noises_labels_path "./labels/data_augmentation_noises_labels.tsv" \
	--augmentation_rirs_labels_path "./labels/data_augmentation_rirs_labels.tsv" \
	--front_end 'VGG' \
	--vgg_n_blocks 3 \
    --vgg_channels [128, 256, 512] \
	--use_weights_and_biases \
	--load_checkpoint \
	--checkpoint_file_folder "./models/24_01_31_11_13_33_VGG_SelfAttention_StatisticalPooling_62eo93za_expert-sunset-20/" \
	--checkpoint_file_name "24_01_31_11_13_33_VGG_SelfAttention_StatisticalPooling_62eo93za_expert-sunset-20.chkpt"