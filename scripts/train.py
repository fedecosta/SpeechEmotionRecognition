# ---------------------------------------------------------------------
# Imports

import argparse
import datetime
import logging
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import f1_score
from torchsummary import summary
import wandb

from data import TrainDataset
from model import Classifier
from loss import FocalLossCriterion
from utils import format_training_labels, generate_model_name, get_memory_info, pad_collate, get_waveforms_stats
from settings import TRAIN_DEFAULT_SETTINGS, LABELS_TO_IDS, LABELS_REDUCED_TO_IDS

# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Logging

# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Classes

class Trainer:

    def __init__(self, input_params):

        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        if input_params.use_weights_and_biases: self.init_wandb()
        self.set_device()
        self.set_random_seed()
        self.set_params(input_params)
        self.set_log_file_handler()
        self.load_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.initialize_training_variables()
        if self.params.use_weights_and_biases: self.config_wandb()
        self.info_mem(logger_level = "DEBUG")
        

    def init_wandb(self):
        
        # Init a wandb project
            
        # TODO fix this, it should be more general to other users
        wandb_run = wandb.init(
            project = "emotions_trains_0", 
            job_type = "training", 
            entity = "upc-veu",
            dir = "/home/usuaris/veussd/federico.costa/logs/wandb/SpeechEmotionRecognition"
            )
        del wandb_run


    def set_device(self):

        '''Set torch device.'''

        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Running on {self.device} device.")
        
        if self.device == "cuda":
            self.gpus_count = torch.cuda.device_count()
            logger.info(f"{self.gpus_count} GPUs available.")
            # Batch size should be divisible by number of GPUs
        else:
            self.gpus_count = 0
        
        logger.info("Device setted.")
    

    def set_random_seed(self):

        logger.info("Setting random seed...")

        random.seed(1234)
        np.random.seed(1234)

        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

        logger.info("Random seed setted.")


    def load_checkpoint(self):

        '''Load trained model checkpoint to continue its training.'''

        # Load checkpoint
        checkpoint_path = os.path.join(
            self.params.checkpoint_file_folder, 
            self.params.checkpoint_file_name,
        )

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        logger.info(f"Checkpoint loaded.")


    def load_checkpoint_params(self):

        '''Load checkpoint original parameters.'''

        logger.info(f"Loading checkpoint params...")

        self.params = self.checkpoint['settings']

        logger.info(f"Checkpoint params loaded.")


    def set_params(self, input_params):

        '''Set Trainer class parameters.'''

        logger.info("Setting params...")
        
        self.params = input_params

        self.params.model_architecture_name = f"{self.params.feature_extractor}_{self.params.front_end}_{self.params.adapter}_{self.params.seq_to_seq_method}_{self.params.seq_to_one_method}"

        if self.params.use_weights_and_biases:
            self.params.model_name = generate_model_name(
            self.params, 
            start_datetime = self.start_datetime, 
            wandb_run_id = wandb.run.id, 
            wandb_run_name = wandb.run.name 
            )
        else:
            self.params.model_name = generate_model_name(
            self.params, 
            start_datetime = self.start_datetime, 
            )


        if self.params.load_checkpoint == True:

            self.load_checkpoint()
            self.load_checkpoint_params()
            # When we load checkpoint params, all input params are overwriten. 
            # So we need to set load_checkpoint flag to True
            self.params.load_checkpoint = True
            # TODO here we could set a new max_epochs value
        
        logger.info("params setted.")


    def set_log_file_handler(self):

        '''Set a logging file handler.'''

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        if self.params.use_weights_and_biases:
            logger_file_name = f"{self.start_datetime}_{wandb.run.id}_{wandb.run.name}.log"
        else:
            logger_file_name = f"{self.start_datetime}.log"
        logger_file_name = logger_file_name.replace(':', '_').replace(' ', '_').replace('-', '_')

        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)

    
    def format_train_labels(self):

        return format_training_labels(
            labels_path = self.params.train_labels_path,
            labels_to_ids = LABELS_REDUCED_TO_IDS,
            prepend_directory = self.params.train_data_dir,
            header = True,
        )

    
    def format_validation_labels(self):

        return format_training_labels(
            labels_path = self.params.validation_labels_path,
            labels_to_ids = LABELS_REDUCED_TO_IDS,
            prepend_directory = self.params.validation_data_dir,
            header = True,
        )


    def format_labels(self):

        '''Return (train_labels_lines, validation_labels_lines)'''

        return self.format_train_labels(), self.format_validation_labels()
         

    def load_training_data(self, train_labels_lines):

        logger.info(f'Loading training data with labels from {self.params.train_labels_path}')

        self.training_wav_mean, self.training_wav_std = get_waveforms_stats(train_labels_lines, self.params.sample_rate)

        # Instanciate a Dataset class
        training_dataset = TrainDataset(
            utterances_paths = train_labels_lines, 
            input_parameters = self.params,
            random_crop_secs = self.params.training_random_crop_secs,
            augmentation_prob = self.params.training_augmentation_prob,
            sample_rate = self.params.sample_rate,
            waveforms_mean = self.training_wav_mean,
            waveforms_std = self.training_wav_std,
            )
        
        # To be used in the weighted loss
        if self.params.weighted_loss:
            self.training_dataset_classes_weights = training_dataset.get_classes_weights()
            self.training_dataset_classes_weights = torch.tensor(self.training_dataset_classes_weights).float().to(self.device)
        
        # Load DataLoader params
        if self.params.text_feature_extractor != 'NoneTextExtractor':
            data_loader_parameters = {
                'batch_size': self.params.training_batch_size, 
                'shuffle': True,
                'num_workers': self.params.num_workers,
                'collate_fn': pad_collate,
                }
        else:
            data_loader_parameters = {
                'batch_size': self.params.training_batch_size, 
                'shuffle': True,
                'num_workers': self.params.num_workers,
                }

        # TODO dont add to the class to get a lighter model?
        # Instanciate a DataLoader class
        self.training_generator = DataLoader(
            training_dataset, 
            **data_loader_parameters,
            )

        del training_dataset

        logger.info("Data and labels loaded.")


    def set_evaluation_batch_size(self):
        # If evaluation is done using the full audio, batch size must be 1 because we will have different-size samples
        if self.params.evaluation_random_crop_secs == 0:
            self.params.evaluation_batch_size = 1


    def load_validation_data(self, validation_labels_lines):

        logger.info(f'Loading data from {self.params.validation_labels_path}')

        # Instanciate a Dataset class
        validation_dataset = TrainDataset(
            utterances_paths = validation_labels_lines, 
            input_parameters = self.params,
            random_crop_secs = self.params.evaluation_random_crop_secs,
            augmentation_prob = self.params.evaluation_augmentation_prob,
            sample_rate = self.params.sample_rate,
            waveforms_mean = self.training_wav_mean,
            waveforms_std = self.training_wav_std,
        )

        # If evaluation_type is total_length, batch size must be 1 because we will have different-size samples
        self.set_evaluation_batch_size()
        
        if self.params.text_feature_extractor != 'NoneTextExtractor':
            data_loader_parameters = {
                'batch_size': self.params.evaluation_batch_size, 
                'shuffle': False,
                'num_workers': self.params.num_workers,
                'collate_fn': pad_collate,
                }
        else:
            data_loader_parameters = {
                'batch_size': self.params.evaluation_batch_size, 
                'shuffle': False,
                'num_workers': self.params.num_workers,
                }
        
        # TODO dont add to the class to get a lighter model?
        # Instanciate a DataLoader class
        self.evaluating_generator = DataLoader(
            validation_dataset, 
            **data_loader_parameters,
            )

        self.evaluation_total_batches = len(self.evaluating_generator)

        del validation_dataset
        
        logger.info("Data and labels loaded.")


    def load_data(self):

        train_labels_lines, validation_labels_lines = self.format_labels()
        self.load_training_data(train_labels_lines)
        self.load_validation_data(validation_labels_lines)
        del train_labels_lines, validation_labels_lines
            

    def load_checkpoint_network(self):

        logger.info(f"Loading checkpoint network...")

        try:
            self.net.load_state_dict(self.checkpoint['model'])
        except RuntimeError:    
            self.net.module.load_state_dict(self.checkpoint['model'])

        logger.info(f"Checkpoint network loaded.")


    def load_network(self):

        # Load the model (Neural Network)

        logger.info("Loading the network...")

        # Load model class
        self.net = Classifier(self.params, self.device)
        
        if self.params.load_checkpoint == True:
            self.load_checkpoint_network()
        
        # Assign model to device
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            # TODO Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel!!!!
            self.net = nn.DataParallel(self.net) 

        logger.info(self.net)

        # Display trainable parameters
        self.total_trainable_params = 0
        parms_dict = {}
        logger.info(f"Detail of every trainable layer:")
        for name, parameter in self.net.named_parameters():

            layer_name = name.split(".")[1]
            if layer_name not in parms_dict.keys():
                parms_dict[layer_name] = 0

            logger.debug(f"name: {name}, layer_name: {layer_name}")

            if not parameter.requires_grad:
                continue
            trainable_params = parameter.numel()

            logger.info(f"{name} is trainable with {parameter.numel()} parameters")
            
            parms_dict[layer_name] = parms_dict[layer_name] + trainable_params
            
            self.total_trainable_params += trainable_params

        logger.info(f"Total trainable parameters per layer:")
        for layer_name in parms_dict.keys():
            logger.info(f"{layer_name}: {parms_dict[layer_name]}")

        #summary(self.net, (150, self.params.feature_extractor_output_vectors_dimension))

        logger.info(f"Network loaded, total_trainable_params: {self.total_trainable_params}")


    def load_loss_function(self):

        logger.info("Loading the loss function...")

        if self.params.loss == "CrossEntropy":
            
            # The nn.CrossEntropyLoss() criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class

            if self.params.weighted_loss:
                logger.info("Using weighted loss function...")
                self.loss_function = nn.CrossEntropyLoss(
                    weight = self.training_dataset_classes_weights,
                )
            else:
                logger.info("Using unweighted loss function...")
                self.loss_function = nn.CrossEntropyLoss()

        elif self.params.loss == "FocalLoss":

            if self.params.weighted_loss:
                logger.info("Using weighted loss function...")
                self.loss_function = FocalLossCriterion(
                    gamma = 2,
                    weights = self.training_dataset_classes_weights,
                )
            else:
                logger.info("Using unweighted loss function...")
                self.loss_function = FocalLossCriterion(
                    gamma = 2,
                )
            
        else:
            raise Exception('No Loss choice found.')  

        logger.info("Loss function loaded.")


    def load_checkpoint_optimizer(self):

        logger.info(f"Loading checkpoint optimizer...")

        self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        logger.info(f"Checkpoint optimizer loaded.")


    def load_optimizer(self):

        logger.info("Loading the optimizer...")

        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )

        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()

        logger.info(f"Optimizer {self.params.optimizer} loaded.")


    def initialize_training_variables(self):

        logger.info("Initializing training variables...")
        
        if self.params.load_checkpoint == True:

            logger.info(f"Loading checkpoint training variables...")

            loaded_training_variables = self.checkpoint['training_variables']

            # HACK this can be refined, but we are going to continue training \
            # from the last epoch trained and from the first batch
            # (even if we may already trained with some batches in that epoch in the last training from the checkpoint).
            self.starting_epoch = loaded_training_variables['epoch']
            self.step = loaded_training_variables['step'] + 1 
            self.validations_without_improvement = loaded_training_variables['validations_without_improvement']
            self.validations_without_improvement_or_opt_update = loaded_training_variables['validations_without_improvement_or_opt_update'] 
            self.early_stopping_flag = False
            self.train_loss = loaded_training_variables['train_loss'] 
            self.training_eval_metric = loaded_training_variables['training_eval_metric'] 
            self.validation_eval_metric = loaded_training_variables['validation_eval_metric'] 
            self.best_train_loss = loaded_training_variables['best_train_loss'] 
            self.best_model_train_loss = loaded_training_variables['best_model_train_loss'] 
            self.best_model_training_eval_metric = loaded_training_variables['best_model_training_eval_metric'] 
            self.best_model_validation_eval_metric = loaded_training_variables['best_model_validation_eval_metric']
            
            logger.info(f"Checkpoint training variables loaded.") 
            logger.info(f"Training will start from:")
            logger.info(f"Epoch {self.starting_epoch}")
            logger.info(f"Step {self.step}")
            logger.info(f"validations_without_improvement {self.validations_without_improvement}")
            logger.info(f"validations_without_improvement_or_opt_update {self.validations_without_improvement_or_opt_update}")
            logger.info(f"Loss {self.train_loss:.3f}")
            logger.info(f"best_model_train_loss {self.best_model_train_loss:.3f}")
            logger.info(f"best_model_training_eval_metric {self.best_model_training_eval_metric:.3f}")
            logger.info(f"best_model_validation_eval_metric {self.best_model_validation_eval_metric:.3f}")

        else:
            self.starting_epoch = 0
            self.step = 0 
            self.validations_without_improvement = 0 
            self.validations_without_improvement_or_opt_update = 0 
            self.early_stopping_flag = False
            self.train_loss = None
            self.training_eval_metric = 0.0
            self.validation_eval_metric = 0.0
            self.best_train_loss = np.inf
            self.best_model_train_loss = np.inf
            self.best_model_training_eval_metric = 0.0
            self.best_model_validation_eval_metric = 0.0
        
        self.total_batches = len(self.training_generator)

        logger.info("Training variables initialized.")


    def config_wandb(self):

        # 1 - Save the params
        self.wandb_config = vars(self.params)

        # 3 - Save additional params

        self.wandb_config["total_trainable_params"] = self.total_trainable_params
        self.wandb_config["gpus"] = self.gpus_count

        # 4 - Update the wandb config
        wandb.config.update(self.wandb_config)


    def evaluate_training(self):

        logger.info(f"Evaluating training task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to("cpu"), torch.tensor([]).to("cpu")
            for batch_number, batch_data in enumerate(self.training_generator):

                if batch_number % 1000 == 0:
                    logger.info(f"Evaluating training task batch {batch_number} of {len(self.training_generator)}...")

                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data      
                else:
                    input, label = batch_data

                # Assign batch data to device
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    transcription_tokens_padded, transcription_tokens_mask = transcription_tokens_padded.long().to(self.device), transcription_tokens_mask.long().to(self.device)
                input, label = input.float().to(self.device), label.long().to(self.device)

                if batch_number == 0: logger.info(f"input.size(): {input.size()}")
                
                # Calculate prediction and loss
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    prediction  = self.net(
                        input_tensor = input, 
                        transcription_tokens_padded = transcription_tokens_padded,
                        transcription_tokens_mask = transcription_tokens_mask,
                        )
                else:
                    prediction  = self.net(input_tensor = input)
                prediction = prediction.to("cpu")
                label = label.to("cpu")

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))
                
            metric_score = f1_score(
                y_true = np.argmax(final_predictions, axis = 1), 
                y_pred = final_labels, 
                average='macro',
                )
            
            self.training_eval_metric = metric_score

            del final_predictions
            del final_labels

        # Return to torch training mode
        self.net.train()

        logger.info(f"Training task evaluated.")
        logger.info(f"F1-score (macro) on training set: {self.training_eval_metric:.3f}")


    def evaluate_validation(self):

        logger.info(f"Evaluating validation task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to("cpu"), torch.tensor([]).to("cpu")
            for batch_number, batch_data in enumerate(self.evaluating_generator):

                if batch_number % 1000 == 0:
                    logger.info(f"Evaluating validation task batch {batch_number} of {len(self.evaluating_generator)}...")

                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data      
                else:
                    input, label = batch_data

                # Assign batch data to device
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    transcription_tokens_padded, transcription_tokens_mask = transcription_tokens_padded.long().to("cpu"), transcription_tokens_mask.long().to("cpu")
                input, label = input.float().to("cpu"), label.long().to("cpu")
                
                if batch_number == 0: logger.info(f"input.size(): {input.size()}")

                # Calculate prediction and loss
                if self.params.text_feature_extractor != 'NoneTextExtractor':
                    prediction  = self.net(
                        input_tensor = input, 
                        transcription_tokens_padded = transcription_tokens_padded,
                        transcription_tokens_mask = transcription_tokens_mask,
                        )
                else:
                    prediction  = self.net(input_tensor = input)
                prediction = prediction.to("cpu")
                label = label.to("cpu")

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))

            metric_score = f1_score(
                y_true = np.argmax(final_predictions, axis = 1), 
                y_pred = final_labels, 
                average='macro',
                )
            
            self.validation_eval_metric = metric_score

            del final_predictions
            del final_labels

        # Return to training mode
        self.net.train()

        logger.info(f"Validation task evaluated.")
        logger.info(f"F1-score (macro) on validation set: {self.validation_eval_metric:.3f}")


    def evaluate(self):

        self.evaluate_training()
        self.evaluate_validation()
             

    def save_model(self):

        '''Function to save the model info and optimizer parameters.'''

        # 1 - Add all the info that will be saved in checkpoint 
        
        model_results = {
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
        }

        training_variables = {
            'epoch': self.epoch,
            'batch_number' : self.batch_number,
            'step' : self.step,
            'validations_without_improvement' : self.validations_without_improvement,
            'validations_without_improvement_or_opt_update' : self.validations_without_improvement_or_opt_update,
            'train_loss' : self.train_loss,
            'training_eval_metric' : self.training_eval_metric,
            'validation_eval_metric' : self.validation_eval_metric,
            'best_train_loss' : self.best_train_loss,
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
            'total_trainable_params' : self.total_trainable_params,
        }
        
        if torch.cuda.device_count() > 1:
            checkpoint = {
                'model': self.net.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }
        else:
            checkpoint = {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }

        end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        checkpoint['start_datetime'] = self.start_datetime
        checkpoint['end_datetime'] = end_datetime

        # 2 - Save the checkpoint locally

        checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
        checkpoint_file_name = f"{self.params.model_name}.chkpt"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

        # Create directory if doesn't exists
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        logger.info(f"Saving training and model information in {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Done.")

        # Delete variables to free memory
        del model_results
        del training_variables
        del checkpoint

        logger.info(f"Training and model information saved.")


    def eval_and_save_best_model(self):

        if self.step > 0 and self.params.eval_and_save_best_model_every > 0 \
            and self.step % self.params.eval_and_save_best_model_every == 0:

            logger.info('Evaluating and saving the new best model (if founded)...')

            # Calculate the evaluation metrics
            self.evaluate()

            # Have we found a better model? (Better in validation metric).
            if self.validation_eval_metric > self.best_model_validation_eval_metric:

                logger.info('We found a better model!')

               # Update best model evaluation metrics
                self.best_model_train_loss = self.train_loss
                self.best_model_training_eval_metric = self.training_eval_metric
                self.best_model_validation_eval_metric = self.validation_eval_metric

                logger.info(f"Best model train loss: {self.best_model_train_loss:.3f}")
                logger.info(f"Best model train evaluation metric: {self.best_model_training_eval_metric:.3f}")
                logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")

                self.save_model() 

                # Since we found and improvement, validations_without_improvement and validations_without_improvement_or_opt_update are reseted.
                self.validations_without_improvement = 0
                self.validations_without_improvement_or_opt_update = 0
            
            else:
                # In this case the search didn't improved the model
                # We are one validation closer to do early stopping
                self.validations_without_improvement = self.validations_without_improvement + 1
                self.validations_without_improvement_or_opt_update = self.validations_without_improvement_or_opt_update + 1
                

            logger.info(f"Consecutive validations without improvement: {self.validations_without_improvement}")
            logger.info(f"Consecutive validations without improvement or optimizer update: {self.validations_without_improvement_or_opt_update}")
            logger.info('Evaluating and saving done.')
            self.info_mem(self.step, logger_level = "DEBUG")


    def check_update_optimizer(self):

        # Update optimizer if neccesary
        if self.validations_without_improvement > 0 and self.validations_without_improvement_or_opt_update > 0\
            and self.params.update_optimizer_every > 0 \
            and self.validations_without_improvement_or_opt_update % self.params.update_optimizer_every == 0:

            if self.params.optimizer == 'sgd' or self.params.optimizer == 'adam' or self.params.optimizer == 'adamw':

                logger.info(f"Updating optimizer...")

                for param_group in self.optimizer.param_groups:

                    param_group['lr'] = param_group['lr'] * self.params.learning_rate_multiplier
                    
                    logger.info(f"New learning rate: {param_group['lr']}")
                
                logger.info(f"Optimizer updated.")

            # We reset validations_without_improvement_or_opt_update since we updated the optimizer
            self.validations_without_improvement_or_opt_update = 0

        # Calculate actual learning rate
        # HACK only taking one param group lr as the overall lr (our case has only one param group)
        for param_group in self.optimizer.param_groups:
            self.learning_rate = param_group['lr']             


    def check_early_stopping(self):

        if self.params.early_stopping > 0 \
            and self.validations_without_improvement >= self.params.early_stopping:

            self.early_stopping_flag = True
            logger.info(f"Doing early stopping after {self.validations_without_improvement} validations without improvement.")

    
    def check_print_training_info(self):
        
        if self.step > 0 and self.params.print_training_info_every > 0 \
            and self.step % self.params.print_training_info_every == 0:

            info_to_print = f"Epoch {self.epoch} of {self.params.max_epochs}, "
            info_to_print = info_to_print + f"batch {self.batch_number} of {self.total_batches}, "
            info_to_print = info_to_print + f"step {self.step}, "
            info_to_print = info_to_print + f"Loss {self.train_loss:.3f}, "
            info_to_print = info_to_print + f"Best validation score: {self.best_model_validation_eval_metric:.3f}..."

            logger.info(info_to_print)

            # Uncomment for memory usage info 
            self.info_mem(self.step, logger_level = "DEBUG")

            
    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch} of {self.params.max_epochs}...")

        # Switch torch to training mode
        self.net.train()

        for self.batch_number, batch_data in enumerate(self.training_generator):

            if self.params.text_feature_extractor != 'NoneTextExtractor':
                input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data  
            else:
                input, label = batch_data

            #logger.info(f"input: {input}")
            #logger.info(f"label: {label}")
            #logger.info(f"transcription_tokens_padded: {transcription_tokens_padded}")
            #logger.info(f"transcription_tokens_mask: {transcription_tokens_mask}")

            # Assign batch data to device
            if self.params.text_feature_extractor != 'NoneTextExtractor':
                transcription_tokens_padded = transcription_tokens_padded.long().to(self.device)
                transcription_tokens_mask = transcription_tokens_mask.long().to(self.device)
    
            input, label = input.float().to(self.device), label.long().to(self.device)
            
            if self.batch_number == 0: logger.info(f"input.size(): {input.size()}")

            # Calculate prediction and loss
            if self.params.text_feature_extractor != 'NoneTextExtractor':
                prediction  = self.net(
                    input_tensor = input, 
                    transcription_tokens_padded = transcription_tokens_padded,
                    transcription_tokens_mask = transcription_tokens_mask,
                    )
            else:
                prediction  = self.net(input_tensor = input)

            self.loss = self.loss_function(prediction, label)
            self.train_loss = self.loss.item()

            # Compute backpropagation and update weights
            
            # Clears x.grad for every parameter x in the optimizer. 
            # It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
            self.optimizer.zero_grad()
            
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. 
            # These are accumulated into x.grad for every parameter x.
            self.loss.backward()
            
            # optimizer.step updates the value of x using the gradient x.grad
            self.optimizer.step()

            # Calculate evaluation metrics and save the best model
            self.eval_and_save_best_model()

            # Update best loss
            if self.train_loss < self.best_train_loss:
                self.best_train_loss = self.train_loss

            self.check_update_optimizer()
            self.check_early_stopping()
            self.check_print_training_info()

            if self.params.use_weights_and_biases:
                try:
                    wandb.log(
                        {
                            "epoch" : self.epoch,
                            "batch_number" : self.batch_number,
                            "loss" : self.train_loss,
                            "learning_rate" : self.learning_rate,
                            "training_eval_metric" : self.training_eval_metric,
                            "validation_eval_metric" : self.validation_eval_metric,
                            'best_model_train_loss' : self.best_model_train_loss,
                            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
                            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
                        },
                        step = self.step
                        )
                except Exception as e:
                    logger.error('Failed at wandb.log: '+ str(e))

            if self.early_stopping_flag == True: 
                break
            
            self.step = self.step + 1

        logger.info(f"-"*50)
        logger.info(f"Epoch {epoch} finished with:")
        logger.info(f"Loss {self.train_loss:.3f}")
        logger.info(f"Best model training evaluation metric: {self.best_model_training_eval_metric:.3f}")
        logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")
        logger.info(f"-"*50)

    
    def train(self, starting_epoch, max_epochs):

        logger.info(f'Starting training for {max_epochs} epochs.')

        for self.epoch in range(starting_epoch, max_epochs):  
            
            self.train_single_epoch(self.epoch)

            if self.early_stopping_flag == True: 
                break
            
        logger.info('Training finished!')


    def delete_version_artifacts(self):

        logger.info(f'Starting to delete not latest checkpoint version artifacts...')

        # We want to keep only the latest checkpoint because of wandb memory storage limit

        api = wandb.Api()
        actual_run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        
        # We need to finish the run and let wandb upload all files
        wandb.run.finish()

        for artifact_version in actual_run.logged_artifacts():
            
            if 'latest' in artifact_version.aliases:
                latest_version = True
            else:
                latest_version = False

            if latest_version == False:
                logger.info(f'Deleting not latest artifact {artifact_version.name} from wandb...')
                artifact_version.delete(delete_aliases=True)
                logger.info(f'Deleted.')

        logger.info(f'All not latest artifacts deleted.')


    def save_model_artifact(self):

        # Save checkpoint as a wandb artifact

        logger.info(f'Starting to save checkpoint as wandb artifact...')

        # Define the artifact
        trained_model_artifact = wandb.Artifact(
            name = self.params.model_name,
            type = "trained_model",
            description = self.params.model_architecture_name,
            metadata = self.wandb_config,
        )

        # Add folder directory
        checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
        logger.info(f'checkpoint_folder {checkpoint_folder}')
        trained_model_artifact.add_dir(checkpoint_folder)

        # Log the artifact
        wandb.run.log_artifact(trained_model_artifact)

        logger.info(f'Artifact saved.')


    def main(self):

        self.train(self.starting_epoch, self.params.max_epochs)
        if self.params.use_weights_and_biases: self.save_model_artifact()
        if self.params.use_weights_and_biases: self.delete_version_artifacts()


    def info_mem(self, step = None, logger_level = "INFO"):

        '''Logs CPU and GPU free memory.'''
        
        cpu_available_pctg, gpu_free = get_memory_info()
        if step is not None:
            message = f"Step {self.step}: CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        else:
            message = f"CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        
        if logger_level == "INFO":
            logger.info(message)
        elif logger_level == "DEBUG":
            logger.debug(message)
#----------------------------------------------------------------------


class ArgsParser:

    def __init__(self):

        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Train a Speech Emotion Recognition model.',
            )


    def add_parser_args(self):
        
        # Directory parameters
        if True:
            
            self.parser.add_argument(
                '--train_labels_path', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['train_labels_path'],
                help = 'Path of the file containing the training examples paths and labels.',
                )
            
            self.parser.add_argument(
                '--train_data_dir', 
                type = str, 
                help = 'Optional additional directory to prepend to the train_labels_path paths.',
                )
            
            self.parser.add_argument(
                '--validation_labels_path', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['validation_labels_path'],
                help = 'Path of the file containing the validation examples paths and labels.',
                )
            
            self.parser.add_argument(
                '--validation_data_dir', 
                type = str, 
                help = 'Optional additional directory to prepend to the validation_labels_path paths.',
                )

            self.parser.add_argument(
                '--augmentation_noises_labels_path', 
                type = str, 
                help = 'Path of the file containing the background noises audio paths and labels.'
                )
            
            self.parser.add_argument(
                '--augmentation_noises_directory', 
                type = str,
                help = 'Optional additional directory to prepend to the augmentation_labels_path paths.',
                )

            self.parser.add_argument(
                '--augmentation_rirs_labels_path', 
                type = str, 
                help = 'Path of the file containing the RIRs audio paths.'
                )
            
            self.parser.add_argument(
                '--augmentation_rirs_directory', 
                type = str, 
                help = 'Optional additional directory to prepend to the rirs_labels_path paths.',
                )

            self.parser.add_argument(
                '--model_output_folder', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['model_output_folder'], 
                help = 'Directory where model outputs and configs are saved.',
                )

            self.parser.add_argument(
                '--checkpoint_file_folder',
                type = str, 
                help = 'Name of folder that contain the model checkpoint file. Mandatory if load_checkpoint is True.',
                )
            
            self.parser.add_argument(
                '--checkpoint_file_name',
                type = str, 
                help = 'Name of the model checkpoint file. Mandatory if load_checkpoint is True.',
                )

            self.parser.add_argument(
                '--log_file_folder',
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['log_file_folder'],
                help = 'Name of folder that will contain the log file.',
                )

        # Data Parameters
        if True:
            
            self.parser.add_argument(
                '--sample_rate', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['sample_rate'],
                help = "Sample rate that you want to use (every audio loaded is resampled to this frequency)."
                )
            
            self.parser.add_argument(
                '--training_random_crop_secs', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['training_random_crop_secs'], 
                help = 'Cut the training input audio with random_crop_secs length at a random starting point. \
                    If 0, the full audio is loaded.'
                )

            self.parser.add_argument(
                '--evaluation_random_crop_secs', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['evaluation_random_crop_secs'], 
                help = 'Cut the evaluation input audio with random_crop_secs length at a random starting point. \
                    If 0, the full audio is loaded.'
                )

            self.parser.add_argument(
                '--num_workers', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['num_workers'],
                help = 'num_workers to be used by the data loader.'
                )
            
            self.parser.add_argument(
                '--padding_type', 
                type = str, 
                choices = ["zero_pad", "repetition_pad"],
                help = 'Type of padding to apply to the audios. \
                    zero_pad does zero left padding, repetition_pad repeats the audio.'
                )
        
        # Data Augmentation arguments
        if True:

            self.parser.add_argument(
                '--training_augmentation_prob', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['training_augmentation_prob'],
                help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
                )

            self.parser.add_argument(
                '--evaluation_augmentation_prob', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['evaluation_augmentation_prob'],
                help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
                )

            self.parser.add_argument(
                '--augmentation_window_size_secs', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['augmentation_window_size_secs'],
                help = 'Cut the audio with augmentation_window_size_secs length at a random starting point. \
                    If 0, the full audio is loaded.'
                )

            self.parser.add_argument(
                '--augmentation_effects', 
                type = str, 
                nargs = '+',
                choices = ["apply_speed_perturbation", "apply_reverb", "add_background_noise"],
                help = 'Effects to augment the data. One or many can be choosen.'
                )
        
        # Network Parameters
        if True:

            self.parser.add_argument(
                '--feature_extractor', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['feature_extractor'],
                choices = ['SpectrogramExtractor', 'WavLMExtractor'], 
                help = 'Type of Feature Extractor used. It should take an audio waveform and output a sequence of vector (features).' 
                )

            self.parser.add_argument(
                '--wavlm_flavor', 
                type = str, 
                choices = ['WAVLM_BASE', 'WAVLM_BASE_PLUS', 'WAVLM_LARGE', 'WAV2VEC2_LARGE_LV60K', 'WAV2VEC2_XLSR_300M', 'WAV2VEC2_XLSR_1B', 'HUBERT_LARGE'], 
                help = 'wavLM model flavor, considered only if WavLMExtractor is used.' 
                )
            
            self.parser.add_argument(
                '--feature_extractor_output_vectors_dimension', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['feature_extractor_output_vectors_dimension'], 
                help = 'Dimension of each vector that will be the output of the feature extractor (usually number of mels in mel-spectrogram).'
                )

            self.parser.add_argument(
                '--text_feature_extractor', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['text_feature_extractor'], 
                choices = ['TextBERTExtractor', 'NoneTextExtractor'], 
                help = 'Type of Text Feature Extractor used. It should take an audio waveform and output a sequence of vector (features).' 
                )

            self.parser.add_argument(
                '--bert_flavor', 
                type = str, 
                choices = ['BERT_BASE_UNCASED', 'BERT_BASE_CASED', 'BERT_LARGE_UNCASED', 'BERT_LARGE_CASED', 'ROBERTA_LARGE'], 
                help = 'BERT model flavor, considered only if TextBERTExtractor is used.' 
                )
            
            self.parser.add_argument(
                '--front_end', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['front_end'],
                choices = ['VGG', 'Resnet34', 'Resnet101', 'NoneFrontEnd'], 
                help = 'Type of Front-end used. \
                    VGG for a N-block VGG architecture.'
                )
            
            self.parser.add_argument(
                '--vgg_n_blocks', 
                type = int, 
                help = 'Number of blocks the VGG front-end block will have.\
                    Each block consists in two convolutional layers followed by a max pooling layer.',
                )

            self.parser.add_argument(
                '--vgg_channels', 
                nargs = '+',
                type = int,
                help = 'Number of channels each VGG convolutional block will have. \
                    The number of channels must be passed in order and consisently with vgg_n_blocks.',
                )
            
            self.parser.add_argument(
                '--adapter', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['adapter'],
                choices = ['NoneAdapter', 'LinearAdapter', 'NonLinearAdapter'], 
                help = 'Type of adapter used.'
                )
            
            self.parser.add_argument(
                '--adapter_output_vectors_dimension', 
                type = int, 
                help = 'Dimension of each vector that will be the output of the adapter layer.',
                )
            
            self.parser.add_argument(
                '--seq_to_seq_method', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['seq_to_seq_method'], 
                choices = ['NoneSeqToSeq', 'SelfAttention', 'MultiHeadAttention', 'TransformerStacked', 'ReducedMultiHeadAttention'], 
                help = 'Sequence to sequence component after the linear projection layer of the model.',
                )

            self.parser.add_argument(
                '--seq_to_seq_heads_number', 
                type = int, 
                help = 'Number of heads for the seq_to_seq layer of the pooling component \
                    (only for MHA based seq_to_seq options).',
                )

            self.parser.add_argument(
                '--transformer_n_blocks', 
                type = int, 
                help = 'Number of transformer blocks to stack in the seq_to_seq component of the pooling. \
                    (Only for seq_to_seq_method = TransformerStacked).',
                )

            self.parser.add_argument(
                '--transformer_expansion_coef', 
                type = int, 
                help = "Number you want to multiply by the size of the hidden layer of the transformer block's feed forward net. \
                    (Only for seq_to_seq_method = TransformerBlock)."
                )
            
            self.parser.add_argument(
                '--transformer_drop_out', 
                type = float, 
                help = 'Dropout probability to use in the feed forward component of the transformer block.\
                    (Only for seq_to_seq_method = TransformerBlock).'
                )
            
            self.parser.add_argument(
                '--seq_to_one_method', 
                type = str, 
                default = TRAIN_DEFAULT_SETTINGS['seq_to_one_method'], 
                choices = ['StatisticalPooling', 'AttentionPooling'], 
                help = 'Type of pooling method applied to the output sequence to sequence component of the model.',
                )

            self.parser.add_argument(
                '--seq_to_seq_input_dropout', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['seq_to_seq_input_dropout'],
                help = 'Dropout probability to use in the seq to seq component input.'
                )

            self.parser.add_argument(
                '--seq_to_one_input_dropout', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['seq_to_one_input_dropout'],
                help = 'Dropout probability to use in the seq to one component input.'
                )
            
            self.parser.add_argument(
                '--classifier_layer_drop_out', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['classifier_layer_drop_out'],
                help = 'Dropout probability to use in the classfifer component.'
                )

            self.parser.add_argument(
                '--classifier_hidden_layers', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['classifier_hidden_layers'],
                help = 'Number of hidden layers in the classifier layer.',
                )

            self.parser.add_argument(
                '--classifier_hidden_layers_width', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['classifier_hidden_layers_width'],
                help = 'Width of every hidden layer in the classifier layer.',
                )
            
            self.parser.add_argument(
                '--number_classes', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['number_classes'],
                help = "Number of classes to classify.",
                )

        # Training Parameters
        if True:

            self.parser.add_argument(
                '--max_epochs',
                type = int,
                default = TRAIN_DEFAULT_SETTINGS['max_epochs'],
                help = 'Max number of epochs to train.',
                )

            self.parser.add_argument(
                '--training_batch_size', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['training_batch_size'],
                help = "Size of training batches.",
                )

            self.parser.add_argument(
                '--evaluation_batch_size', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['evaluation_batch_size'],
                help = "Size of evaluation batches.",
                )

            self.parser.add_argument(
                '--eval_and_save_best_model_every', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['eval_and_save_best_model_every'],
                help = "The model is evaluated on train and validation sets and saved every eval_and_save_best_model_every steps. \
                    Set to 0 if you don't want to execute this utility.",
                )
            
            self.parser.add_argument(
                '--print_training_info_every', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['print_training_info_every'],
                help = "Training info is printed every print_training_info_every steps. \
                    Set to 0 if you don't want to execute this utility.",
                )

            self.parser.add_argument(
                '--early_stopping', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['early_stopping'],
                help = "Training is stopped if there are early_stopping consectuive validations without improvement. \
                    Set to 0 if you don't want to execute this utility.",
                )

            self.parser.add_argument(
                '--load_checkpoint',
                action = argparse.BooleanOptionalAction,
                default = TRAIN_DEFAULT_SETTINGS['load_checkpoint'],
                help = 'Set to True if you want to load a previous checkpoint and continue training from that point. \
                    Loaded parameters will overwrite all inputted parameters.',
                )

        # Optimization arguments
        if True:
            
            self.parser.add_argument(
                '--optimizer', 
                type = str, 
                choices = ['adam', 'sgd', 'rmsprop', 'adamw'], 
                default = TRAIN_DEFAULT_SETTINGS['optimizer'],
                )

            self.parser.add_argument(
                '--learning_rate', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['learning_rate'],
                )
            
            self.parser.add_argument(
                '--learning_rate_multiplier', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['learning_rate_multiplier'],
                )

            self.parser.add_argument(
                '--weight_decay', 
                type = float, 
                default = TRAIN_DEFAULT_SETTINGS['weight_decay'],
                )
            
            self.parser.add_argument(
                '--update_optimizer_every', 
                type = int, 
                default = TRAIN_DEFAULT_SETTINGS['update_optimizer_every'],
                help = "Some optimizer parameters will be updated every update_optimizer_every consecutive validations without improvement. \
                    Set to 0 if you don't want to execute this utility.",
                )

            self.parser.add_argument(
                '--loss', 
                type = str, 
                choices = ['CrossEntropy', 'FocalLoss'], 
                default = TRAIN_DEFAULT_SETTINGS['loss'],
                )
            
            self.parser.add_argument(
                "--weighted_loss", 
                action = argparse.BooleanOptionalAction,
                default = TRAIN_DEFAULT_SETTINGS['weighted_loss'],
                help = "Set the weight parameter of the loss to a tensor representing the inverse frequency of each class.",
                )


        # Verbosity and debug Parameters
        if True:
            
            self.parser.add_argument(
                "--use_weights_and_biases", 
                action = argparse.BooleanOptionalAction,
                default = TRAIN_DEFAULT_SETTINGS['use_weights_and_biases'],
                help = "Use weights and Biases.",
                )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()

# # --------------------------------------------------------------------- 

if __name__ == "__main__":

    args_parser = ArgsParser()
    args_parser.main()
    trainer_parameters = args_parser.arguments

    trainer = Trainer(trainer_parameters)
    trainer.main()