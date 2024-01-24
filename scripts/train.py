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
from torcheval.metrics.functional import multiclass_f1_score
import torchaudio
from torchsummary import summary

if False:
    import pandas as pd
    import pickle
    

from data import TrainDataset
from model import Classifier
from utils import format_training_labels, generate_model_name
from settings import TRAIN_DEFAULT_SETTINGS

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
        self.set_device()
        self.set_random_seed()
        self.set_params(input_params)
        self.set_log_file_handler()
        self.load_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.initialize_training_variables()
        # self.config_wandb()


    def set_device(self):

        '''Set torch device.'''

        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Running on {self.device} device.")
        
        if self.device == "cuda":
            self.gpus_count = torch.cuda.device_count()
            logger.info(f"{self.gpus_count} GPUs available.")
        
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
        checkpoint_folder = self.params.checkpoint_file_folder
        checkpoint_file_name = self.params.checkpoint_file_name
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

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
        self.params.model_name = generate_model_name(
           self.params, 
           start_datetime = self.start_datetime, 
           #wandb_run_id = wandb.run.id, # FIX when setting wandb
           #wandb_run_name = wandb.run.name # FIX when setting wandb
           )

        if self.params.load_checkpoint == True:

            self.load_checkpoint()
            self.load_checkpoint_params()
            # When we load checkpoint params, all input params are overwriten. 
            # So we need to set load_checkpoint flag to True
            self.params.load_checkpoint = True
        
        logger.info("params setted.")


    def set_log_file_handler(self):

        '''Set a logging file handler.'''

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        logger_file_name = f"{self.params.model_name}.log"
        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)

    
    def format_train_labels(self):

        self.train_labels_lines = format_training_labels(
            labels_path = self.params.train_labels_path,
            prepend_directory = self.params.train_data_dir,
            header = True,
        )

    
    def format_validation_labels(self):

        self.validation_labels_lines = format_training_labels(
            labels_path = self.params.validation_labels_path,
            prepend_directory = self.params.validation_data_dir,
            header = True,
        )


    def format_labels(self):

        self.format_train_labels()
        self.format_validation_labels()
         

    def load_training_data(self):

        logger.info(f'Loading training data with labels from {self.params.train_labels_path}')

        # Instanciate a Dataset class
        training_dataset = TrainDataset(
            utterances_paths = self.train_labels_lines, 
            input_parameters = self.params,
            random_crop_secs = self.params.training_random_crop_secs,
            )
        
        # Load DataLoader params
        data_loader_parameters = {
            'batch_size': self.params.training_batch_size, 
            'shuffle': True, # FIX hardcoded True
            'num_workers': self.params.num_workers,
            }
        
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


    def load_validation_data(self):

        logger.info(f'Loading data from {self.params.validation_labels_path}')

        # Instanciate a Dataset class
        validation_dataset = TrainDataset(
            utterances_paths = self.validation_labels_lines, 
            input_parameters = self.params,
            random_crop_secs = self.params.evaluation_random_crop_secs,
        )

        # If evaluation_type is total_length, batch size must be 1 because we will have different-size samples
        self.set_evaluation_batch_size()
        
        # Instanciate a DataLoader class
        self.evaluating_generator = DataLoader(
            validation_dataset, 
            batch_size = self.params.evaluation_batch_size,
            shuffle = False,
            num_workers = self.params.num_workers,
            )

        self.evaluation_total_batches = len(self.evaluating_generator)

        del validation_dataset
        
        logger.info("Data and labels loaded.")


    def load_data(self):

        self.format_labels()
        self.load_training_data()
        self.load_validation_data()
            

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
        
        summary(self.net, (350, self.params.front_end_input_vectors_dim))

        # Calculate trainable parameters (to estimate model complexity)
        self.total_trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )

        logger.info("Network loaded.")


    def load_loss_function(self):

        logger.info("Loading the loss function...")

        # The nn.CrossEntropyLoss() criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        self.loss_function = nn.CrossEntropyLoss()

        logger.info("Loss function loaded.")


    def load_checkpoint_optimizer(self):

        logger.info(f"Loading checkpoint optimizer...")

        self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        logger.info(f"Checkpoint optimizer loaded.")


    def load_optimizer(self):

        logger.info("Loading the optimizer...")

        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )
        if self.params.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
                )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.net.parameters(), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay
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

        if False:

            # Init a wandb project
            import wandb
            run = wandb.init(
                project = "emotions_models_1", 
                job_type = "training", 
                entity = "upc-veu",
                )

            # 1 - Save the params
            self.wandb_config = vars(self.params)

            # 3 - Save additional params

            self.wandb_config["total_trainable_params"] = self.total_trainable_params
            self.wandb_config["gpus"] = self.gpus

            # 4 - Update the wandb config
            wandb.config.update(self.wandb_config)


    def evaluate_training(self):

        logger.info(f"Evaluating training task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            for batch_number, (input, label) in enumerate(self.training_generator):

                # Assign input and label to device
                input, label = input.float().to(self.device), label.long().to(self.device)

                # Calculate prediction and loss
                prediction  = self.net(input_tensor = input, label = label)

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))

            metric_score = multiclass_f1_score(
                input = final_predictions, 
                target  = final_labels, 
                num_classes = self.params.number_classes,
                average = 'micro', # TODO think what method is best to define
                )
            
            self.training_eval_metric = metric_score

            # TODO maybe we should need to clean memory

        # Return to torch training mode
        self.net.train()

        logger.info(f"Training task evaluated.")
        logger.info(f"F1-score (micro) on training set: {self.training_eval_metric:.3f}")


    def evaluate_validation(self):

        logger.info(f"Evaluating validation task...")

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            final_predictions, final_labels = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            for batch_number, (input, label) in enumerate(self.evaluating_generator):

                # Assign input and label to device
                input, label = input.float().to(self.device), label.long().to(self.device)

                # Calculate prediction and loss
                prediction  = self.net(input_tensor = input, label = label)

                final_predictions = torch.cat(tensors = (final_predictions, prediction))
                final_labels = torch.cat(tensors = (final_labels, label))

            # TODO complete this
            metric_score = multiclass_f1_score(
                input = final_predictions, 
                target  = final_labels, 
                num_classes = self.params.number_classes,
                average = 'micro', # TODO think what method is best to define
                )
            
            self.validation_eval_metric = metric_score

            # TODO maybe we should need to clean memory

        # Return to training mode
        self.net.train()

        logger.info(f"Validation task evaluated.")
        logger.info(f"F1-score (micro) on validation set: {self.validation_eval_metric:.3f}")


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


    def check_update_optimizer(self):

        # Update optimizer if neccesary
        if self.validations_without_improvement > 0 and self.validations_without_improvement_or_opt_update > 0\
            and self.params.update_optimizer_every > 0 \
            and self.validations_without_improvement_or_opt_update % self.params.update_optimizer_every == 0:

            if self.params.optimizer == 'sgd' or self.params.optimizer == 'adam':

                logger.info(f"Updating optimizer...")

                for param_group in self.optimizer.param_groups:

                    param_group['lr'] = param_group['lr'] * 0.5 # FIX hardcoded value  
                    
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

            
    def train_single_epoch(self, epoch):

        logger.info(f"Epoch {epoch} of {self.params.max_epochs}...")

        # Switch torch to training mode
        self.net.train()

        for self.batch_number, (input, label) in enumerate(self.training_generator):

            # Assign input and label to device
            input, label = input.float().to(self.device), label.long().to(self.device)

            # Calculate prediction and loss
            prediction  = self.net(input_tensor = input, label = label)

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

            if False:
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

        if False:

            logger.info(f'Starting to delete not latest checkpoint version artifacts...')

            # We want to keep only the latest checkpoint because of wandb memory storage limit

            api = wandb.Api()
            actual_run = api.run(f"{run.entity}/{run.project}/{run.id}")
            
            # We need to finish the run and let wandb upload all files
            run.finish()

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

        if False:

            # Save checkpoint as a wandb artifact

            logger.info(f'Starting to save checkpoint as wandb artifact...')

            # Define the artifact
            trained_model_artifact = wandb.Artifact(
                name = self.params.model_name,
                type = "trained_model",
                description = self.params.model_name_prefix, # TODO set as an argparse input param
                metadata = self.wandb_config,
            )

            # Add folder directory
            checkpoint_folder = os.path.join(self.params.model_output_folder, self.params.model_name)
            logger.info(f'checkpoint_folder {checkpoint_folder}')
            trained_model_artifact.add_dir(checkpoint_folder)

            # Log the artifact
            run.log_artifact(trained_model_artifact)

            logger.info(f'Artifact saved.')


    def main(self):

        self.train(self.starting_epoch, self.params.max_epochs)
        #self.save_model_artifact()

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
        # ---------------------------------------------------------------------

        self.parser.add_argument(
            '--train_labels_path', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['train_labels_path'],
            help = 'Path of the file containing the training examples paths and labels.',
            )
        
        self.parser.add_argument(
            '--train_data_dir', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['train_data_dir'],
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
            default = TRAIN_DEFAULT_SETTINGS['validation_data_dir'],
            help = 'Optional additional directory to prepend to the validation_labels_path paths.',
            )

        self.parser.add_argument(
            '--model_output_folder', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['model_output_folder'], 
            help = 'Directory where model outputs and configs are saved.',
            )

        self.parser.add_argument(
            '--log_file_folder',
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['log_file_folder'],
            help = 'Name of folder that will contain the log file.',
            )
#         # ---------------------------------------------------------------------

#         # Training Parameters
#         # ---------------------------------------------------------------------
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
            '--update_optimizer_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['update_optimizer_every'],
            help = "Some optimizer parameters will be updated every update_optimizer_every consecutive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
            )

        self.parser.add_argument(
            '--load_checkpoint',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['load_checkpoint'],
            help = 'Set to True if you want to load a previous checkpoint and continue training from that point. \
                Loaded parameters will overwrite all inputted parameters.',
            )

        self.parser.add_argument(
            '--checkpoint_file_folder',
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['checkpoint_file_folder'],
            help = 'Name of folder that contain the model checkpoint file. Mandatory if load_checkpoint is True.',
            )
        
        self.parser.add_argument(
            '--checkpoint_file_name',
            type = str, 
            help = 'Name of the model checkpoint file. Mandatory if load_checkpoint is True.',
            )
        
#         # ---------------------------------------------------------------------

#         # Evaluation Parameters
#         # ---------------------------------------------------------------------
#         self.parser.add_argument(
#             '--evaluation_type', 
#             type = str, 
#             choices = ['random_crop', 'total_length'],
#             default = TRAIN_DEFAULT_SETTINGS['evaluation_type'], 
#             help = 'With random_crop the utterances are croped at random with random_crop_secs secs before doing the forward pass.\
#                 In this case, samples are batched with batch_size.\
#                 With total_length, full length audios are passed through the forward.\
#                 In this case, samples are automatically batched with batch_size = 1, since they have different lengths.',
#             )

#         self.parser.add_argument(
#             '--evaluation_batch_size', 
#             type = int, 
#             default = TRAIN_DEFAULT_SETTINGS['evaluation_batch_size'],
#             help = "Size of evaluation batches. Automatically set to 1 if evaluation_type is total_length.",
#             )
#         # ---------------------------------------------------------------------

#         # Data Parameters
#         # ---------------------------------------------------------------------
        
        
        
        self.parser.add_argument(
            '--front_end_input_vectors_dim', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['front_end_input_vectors_dim'], 
            help = 'Dimension of each vector that will be input of the front-end (usually number of mels in mel-spectrogram).'
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

#         self.parser.add_argument(
#             '--normalization', 
#             type = str, 
#             default = TRAIN_DEFAULT_SETTINGS['normalization'], 
#             choices = ['cmn', 'cmvn', 'full'],
#             help = 'Type of normalization applied to the features when evaluating in validation. \
#                 It can be Cepstral Mean Normalization or Cepstral Mean and Variance Normalization',
#             )

        self.parser.add_argument(
            '--num_workers', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['num_workers'],
            help = 'num_workers to be used by the data loader.'
            )
#         # ---------------------------------------------------------------------
        
#         # Network Parameters
#         # ---------------------------------------------------------------------
        
        self.parser.add_argument(
            '--number_classes', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['number_classes'],
            help = "Number of classes to classify.",
            )

#         self.parser.add_argument(
#             '--embedding_size', 
#             type = int, 
#             default = TRAIN_DEFAULT_SETTINGS['embedding_size'],
#             help = 'Size of the embedding that the system will generate.',
#             )

        self.parser.add_argument(
            '--front_end', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['front_end'],
            choices = ['VGG',], 
            help = 'Type of Front-end used. \
                VGG for a N-block VGG architecture.'
            )
            
        self.parser.add_argument(
            '--vgg_n_blocks', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['vgg_n_blocks'],
            help = 'Number of blocks the VGG front-end block will have.\
                Each block consists in two convolutional layers followed by a max pooling layer.',
            )

        self.parser.add_argument(
            '--vgg_channels', 
            nargs = '+',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['vgg_channels'],
            help = 'Number of channels each VGG convolutional block will have. \
                The number of channels must be passed in order and consisently with vgg_n_blocks.',
            )

#         self.parser.add_argument(
#             '--patchs_generator_patch_width', 
#             type = int,
#             help = 'Width of each patch token to use with at the PatchsGenerator front-end. \
#                 (Only used when front_end = PatchsGenerator.',
#             )

        self.parser.add_argument(
            '--pooling_method', 
            type = str, 
            default = TRAIN_DEFAULT_SETTINGS['pooling_method'], 
            choices = ['StatisticalPooling',], 
            help = 'Type of pooling method.',
            )

        self.parser.add_argument(
            '--pooling_output_size', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['pooling_output_size'], 
            help = 'Each output vector of the pooling component will have 1 x pooling_output_size dimension.',
            )

#         self.parser.add_argument(
#             '--pooling_heads_number', 
#             type = int, 
#             #default = TRAIN_DEFAULT_SETTINGS['pooling_heads_number'],
#             help = 'Number of heads for the attention layer of the pooling component \
#                 (only for MHA based pooling_method options).',
#             )

#         self.parser.add_argument(
#             '--pooling_mask_prob', 
#             type = float, 
#             #default = TRAIN_DEFAULT_SETTINGS['pooling_mask_prob'], 
#             help = 'Masking head drop probability. Only used for pooling_method = Double MHA',
#             )

#         self.parser.add_argument(
#             '--pooling_positional_encoding', 
#             action = argparse.BooleanOptionalAction,
#             #default = TRAIN_DEFAULT_SETTINGS['pooling_positional_encoding'], 
#             help = 'Wether to use positional encoding in the attention layer of the pooling component.'
#             )

#         self.parser.add_argument(
#             '--transformer_n_blocks', 
#             type = int, 
#             #default = TRAIN_DEFAULT_SETTINGS['transformer_n_blocks'],
#             help = 'Number of transformer blocks to stack in the attention component of the pooling_method. \
#                 (Only for pooling_method = TransformerStackedAttentionPooling).',
#             )

#         self.parser.add_argument(
#             '--transformer_expansion_coef', 
#             type = int, 
#             #default = TRAIN_DEFAULT_SETTINGS['transformer_expansion_coef'], 
#             help = "Number you want to multiply by the size of the hidden layer of the transformer block's feed forward net. \
#                 (Only for pooling_method = TransformerStackedAttentionPooling)."
#             )

#         self.parser.add_argument(
#             '--transformer_attention_type', 
#             type = str, 
#             #default = TRAIN_DEFAULT_SETTINGS['transformer_attention_type'], 
#             choices = ['SelfAttention', 'MultiHeadAttention'],
#             help = 'Type of Attention to use in the attention component of the transformer block.\
#                 (Only for pooling_method = TransformerStackedAttentionPooling).'
#             )
        
#         self.parser.add_argument(
#             '--transformer_drop_out', 
#             type = float, 
#             #default = TRAIN_DEFAULT_SETTINGS['transformer_drop_out'], 
#             help = 'Dropout probability to use in the feed forward component of the transformer block.\
#                 (Only for pooling_method = TransformerStackedAttentionPooling).'
#             )

#         self.parser.add_argument(
#             '--bottleneck_drop_out', 
#             type = float, 
#             default = TRAIN_DEFAULT_SETTINGS['bottleneck_drop_out'], 
#             help = 'Dropout probability to use in each layer of the final fully connected bottleneck.'
#             )
#         # ---------------------------------------------------------------------

#         # AMSoftmax Config
#         # ---------------------------------------------------------------------
#         self.parser.add_argument(
#             '--scaling_factor', 
#             type = float, 
#             default = TRAIN_DEFAULT_SETTINGS['scaling_factor'], 
#             help = 'Scaling factor of the AM-Softmax (referred as s in the AM-Softmax definition).'
#             )

#         self.parser.add_argument(
#             '--margin_factor', 
#             type = float, 
#             default = TRAIN_DEFAULT_SETTINGS['margin_factor'],
#             help = 'Margin factor of the AM-Softmax (referred as m in the AM-Softmax definition).'
#             )
#         # ---------------------------------------------------------------------

        # Optimization arguments
        # ---------------------------------------------------------------------
        self.parser.add_argument(
            '--optimizer', 
            type = str, 
            choices = ['adam', 'sgd', 'rmsprop'], 
            default = TRAIN_DEFAULT_SETTINGS['optimizer'],
            )

        self.parser.add_argument(
            '--learning_rate', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['learning_rate'],
            )

        self.parser.add_argument(
            '--weight_decay', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['weight_decay'],
            )
        # ---------------------------------------------------------------------

        # Verbosity and debug Parameters
        # ---------------------------------------------------------------------
#         self.parser.add_argument(
#             "--verbose", 
#             action = argparse.BooleanOptionalAction,
#             default = TRAIN_DEFAULT_SETTINGS['verbose'],
#             help = "Increase output verbosity.",
#             )


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