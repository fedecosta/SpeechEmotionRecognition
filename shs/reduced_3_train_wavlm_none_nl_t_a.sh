#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_reduced_3
python scripts/train.py \
	--train_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--validation_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--train_labels_path './labels/training_labels_reduced_8_classes.tsv' \
	--validation_labels_path './labels/development_labels_reduced_8_classes.tsv' \
	--augmentation_noises_labels_path "./labels/data_augmentation_noises_labels.tsv" \
	--augmentation_rirs_labels_path "./labels/data_augmentation_rirs_labels.tsv" \
	--training_random_crop_secs 3.5 \
	--evaluation_random_crop_secs 3.5 \
	--augmentation_window_size_secs 3.5 \
	--training_augmentation_prob 0.75 \
	--evaluation_augmentation_prob 0 \
	--feature_extractor 'WavLMExtractor' \
	--feature_extractor_output_vectors_dimension 768 \
	--front_end 'NoneFrontEnd' \
	--adapter 'NonLinearAdapter' \
	--adapter_output_vectors_dimension 256 \
	--seq_to_seq_method 'TransformerStacked' \
	--seq_to_seq_heads_number 4 \
	--transformer_n_blocks 2 \
	--transformer_drop_out 0 \
	--transformer_expansion_coef 4 \
	--seq_to_one_method 'AttentionPooling' \
	--max_epochs 200 \
	--training_batch_size 32 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 900 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 4 \
	--number_classes 8 \
	--weighted_loss \
	--use_weights_and_biases