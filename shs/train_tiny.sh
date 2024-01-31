#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
python scripts/train.py \
	--train_labels_path "./labels/training_labels_tiny.tsv" \
	--validation_labels_path "./labels/development_labels_tiny.tsv" \
	--augmentation_noises_labels_path "./labels/data_augmentation_noises_labels.tsv" \
	--augmentation_rirs_labels_path "./labels/data_augmentation_rirs_labels.tsv" \
	--augmentation_window_size_secs 2.0 \
	--training_batch_size 1 \
	--print_training_info_every 1 \
	--eval_and_save_best_model_every 9 \
	--max_epochs 5 \
	--use_weights_and_biases