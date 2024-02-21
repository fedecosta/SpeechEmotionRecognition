#!/bin/bash
#SBATCH -o /home/usuaris/veussd/federico.costa/logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_8_classes_6
python scripts/train.py \
	--train_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--validation_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files' \
	--train_labels_path './labels/training_labels_reduced_8_classes.tsv' \
	--validation_labels_path './labels/development_labels_reduced_8_classes.tsv' \
	--augmentation_noises_labels_path "./labels/data_augmentation_noises_labels.tsv" \
	--augmentation_rirs_labels_path "./labels/data_augmentation_rirs_labels.tsv" \
	--model_output_folder "/home/usuaris/veussd/federico.costa/models/" \
	--log_file_folder "/home/usuaris/veussd/federico.costa/logs/train/" \
	--training_random_crop_secs 5.5 \
	--evaluation_random_crop_secs 0 \
	--augmentation_window_size_secs 5.5 \
	--training_augmentation_prob 0.5 \
	--evaluation_augmentation_prob 0 \
	--augmentation_effects 'apply_speed_perturbation' 'apply_reverb' 'add_background_noise' \
	--feature_extractor 'WavLMExtractor' \
	--feature_extractor_output_vectors_dimension 768 \
	--text_feature_extractor 'TextBERTExtractor' \
	--front_end 'NoneFrontEnd' \
	--adapter 'NoneAdapter' \
	--seq_to_seq_method 'MultiHeadAttention' \
	--seq_to_seq_heads_number 4 \
	--seq_to_one_method 'AttentionPooling' \
	--max_epochs 200 \
	--training_batch_size 32 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 3100 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 4 \
	--padding_type 'repetition_pad' \
	--classifier_layer_drop_out 0.1 \
	--number_classes 8 \
	--weighted_loss \
	--learning_rate_multiplier 0.9 \
	--use_weights_and_biases