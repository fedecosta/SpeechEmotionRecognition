LABELS_TO_IDS = {
    "n": 0,
    "x": 1,
    "h": 2,
    "a": 3,
    "c": 4,
    "s": 5,
    "u": 6,
    "d": 7,
    "o": 8,
    "f": 9,
}

LABELS_REDUCED_TO_IDS = {
    "h": 0,
    "a": 1,
    "c": 2,
    "s": 3,
    "u": 4,
    "d": 5,
    "f": 6,
    }

TRAIN_DEFAULT_SETTINGS = {
    'train_labels_path' : './labels/training_labels.tsv',
    'validation_labels_path' : './labels/development_labels.tsv',
    'model_output_folder' : './models/',
    'log_file_folder' : './logs/train/',
    'sample_rate': 16000,
    'training_random_crop_secs' : 2.0,
    'evaluation_random_crop_secs' : 2.0,
    'num_workers' : 0,
    'training_augmentation_prob' : 0.75,
    'evaluation_augmentation_prob' : 0,
    'augmentation_window_size_secs' : 2.0,
    'feature_extractor': 'SpectrogramExtractor',
    'feature_extractor_output_vectors_dimension' : 80,
    'front_end' : 'NoneFrontEnd',
    'adapter': 'NoneAdapter',
    'seq_to_seq_method' : 'NoneSeqToSeq',
    'seq_to_one_method' : 'StatisticalPooling',
    'classifier_layer_drop_out' : 0,
    'number_classes' : 10,
    'max_epochs' : 100,
    'training_batch_size' : 64,
    'evaluation_batch_size' : 1,
    'eval_and_save_best_model_every' : 1000,
    'print_training_info_every' : 100,
    'early_stopping' : 25,
    'load_checkpoint' : False,
    'optimizer' : 'adam',
    'learning_rate' : 0.0001,
    'learning_rate_multiplier' : 0.5,
    'weight_decay' : 0.001,
    'update_optimizer_every' : 10,
    'weighted_loss' : True,
    'use_weights_and_biases' : False,
}

