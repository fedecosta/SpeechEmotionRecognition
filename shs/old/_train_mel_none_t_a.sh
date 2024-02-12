#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_1
python scripts/train.py \
	--train_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--validation_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--augmentation_noises_labels_path "./labels/data_augmentation_noises_labels.tsv" \
	--augmentation_rirs_labels_path "./labels/data_augmentation_rirs_labels.tsv" \
	--training_random_crop_secs 2.0 \
	--evaluation_random_crop_secs 0.0 \
	--augmentation_window_size_secs 2.0 \
	--training_augmentation_prob 0.75 \
	--evaluation_augmentation_prob 0 \
	--feature_extractor 'Spectrogram' \
	--feature_extractor_output_vectors_dimension 80 \
	--front_end 'NoneFrontEnd' \
	--adapter_output_vectors_dimension 64 \
	--seq_to_seq_method 'TransformerStacked' \
	--seq_to_seq_heads_number 2 \
	--transformer_n_blocks 2 \
	--transformer_expansion_coef 4 \
	--transformer_drop_out 0.1 \
	--seq_to_one_method 'AttentionPooling' \
	--max_epochs 100 \
	--training_batch_size 32 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 2000 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 8 \
	--no-use_weights_and_biases