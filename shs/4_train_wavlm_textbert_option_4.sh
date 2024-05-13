#!/bin/bash
#SBATCH -o /home/usuaris/veussd/federico.costa/logs/sbatch/outputs/slurm-%j.out
#SBATCH -e /home/usuaris/veussd/federico.costa/logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu            # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_8_classes_4
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
	--wavlm_flavor 'WAV2VEC2_XLSR_300M' \
	--feature_extractor_output_vectors_dimension 1024 \
	--text_feature_extractor 'TextBERTExtractor' \
	--bert_flavor 'BERT_LARGE_UNCASED' \
	--front_end 'NoneFrontEnd' \
	--adapter 'NoneAdapter' \
	--seq_to_seq_method 'MultiHeadAttention' \
	--seq_to_seq_heads_number 4 \
	--seq_to_seq_input_dropout 0.0 \
	--seq_to_one_method 'AttentionPooling' \
	--seq_to_one_input_dropout 0.0 \
	--max_epochs 20 \
	--training_batch_size 32 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 1600 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 4 \
	--padding_type 'repetition_pad' \
	--classifier_hidden_layers 4 \
	--classifier_hidden_layers_width 512 \
	--classifier_layer_drop_out 0.1 \
	--number_classes 8 \
	--loss 'FocalLoss' \
	--no-weighted_loss \
	--optimizer 'adamw' \
	--update_optimizer_every 5 \
	--learning_rate 0.0001 \
	--learning_rate_multiplier 0.5 \
	--weight_decay 0.01 \
	--use_weights_and_biases