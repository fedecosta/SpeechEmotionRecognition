from torch.utils import data
import numpy as np
import torchaudio
import torch
from random import randint
import copy
from augmentation import DataAugmentator
import random
import logging
import pandas as pd
from text_feature_extractor import ASRModel, ASRDummy

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

class TrainDataset(data.Dataset):

    def __init__(self, utterances_paths, input_parameters, random_crop_secs, augmentation_prob = 0, sample_rate = 16000, waveforms_mean = None, waveforms_std = None):
        
        self.utterances_paths = utterances_paths
        # I suspect when instantiating two datasets the parameters are overrided
        # TODO maybe avoid defining self.parameters to reduce memory usage
        self.parameters = copy.deepcopy(input_parameters) 
        self.augmentation_prob = augmentation_prob
        self.sample_rate = sample_rate
        self.waveforms_mean = waveforms_mean
        self.waveforms_std = waveforms_std
        self.random_crop_secs = random_crop_secs
        self.random_crop_samples = int(self.random_crop_secs * self.sample_rate)
        self.num_samples = len(utterances_paths)
        if self.augmentation_prob > 0: self.init_data_augmentator()
        if self.parameters.text_feature_extractor != 'NoneTextExtractor': self.init_transcriptor()


    def get_classes_weights(self):

        dataset_labels = [path.strip().split('\t')[1] for path in self.utterances_paths]

        weights_series = pd.Series(dataset_labels).value_counts(normalize = True, dropna = False)
        weights_df = pd.DataFrame(weights_series).reset_index()
        weights_df.columns = ["class_id", "weight"]
        weights_df["weight"] = 1 / weights_df["weight"]
        weights_df = weights_df.sort_values("class_id", ascending=True)
        
        weights = weights_df["weight"].to_list()

        for class_id in range(len(weights)):
            logger.info(f"Class_id {class_id} weight: {weights[class_id]}")
        
        return weights


    def init_data_augmentator(self):

        self.data_augmentator = DataAugmentator(
            self.parameters.augmentation_noises_directory,
            self.parameters.augmentation_noises_labels_path,
            self.parameters.augmentation_rirs_directory,
            self.parameters.augmentation_rirs_labels_path,
            self.parameters.augmentation_window_size_secs,
            self.parameters.augmentation_effects,
        )


    def init_transcriptor(self):

        # HACK we need to transcribe into this class because the ASR model has a problem implementing batches transcriptions
        
        self.transcriptor = ASRDummy()

        if self.parameters.bert_flavor == "BERT_BASE_UNCASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        elif self.parameters.bert_flavor == "BERT_BASE_CASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        elif self.parameters.bert_flavor == "BERT_LARGE_UNCASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased')
        elif self.parameters.bert_flavor == "BERT_LARGE_CASED":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-cased')
        elif self.parameters.bert_flavor == "ROBERTA_LARGE":
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'roberta-large')
        else:
            raise Exception('No bert_flavor choice found.')
    

    def pad_waveform(self, waveform, padding_type, random_crop_samples):

        if padding_type == "zero_pad":
            pad_left = max(0, self.random_crop_samples - waveform.shape[-1])
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, 0), mode = "constant")
        elif padding_type == "repetition_pad":
            necessary_repetitions = int(np.ceil(random_crop_samples / waveform.size(-1)))
            padded_waveform = waveform.repeat(necessary_repetitions)
        else:
            raise Exception('No padding choice found.') 
        
        return padded_waveform
        
    
    def sample_audio_window(self, waveform, random_crop_samples):

        waveform_total_samples = waveform.size()[-1]
        
        # TODO maybe we can add an assert to check that random_crop_samples <= waveform_total_samples (will it slow down the process?)
        random_start_index = randint(0, waveform_total_samples - random_crop_samples)
        end_index = random_start_index + random_crop_samples
        
        cropped_waveform =  waveform[random_start_index : end_index]

        return cropped_waveform

    
    def normalize(self, waveform):

        if self.waveforms_mean is not None and self.waveforms_std is not None:
            normalized_waveform = (waveform - self.waveforms_mean) / (self.waveforms_std + 0.000001)
        else:
            normalized_waveform = waveform

        return normalized_waveform
    
    
    def process_waveform(self, waveform, original_sample_rate):

        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform = waveform,
                orig_freq = original_sample_rate, 
                new_freq = self.sample_rate, 
                )

        if random.uniform(0, 0.999) > 1 - self.augmentation_prob:
            waveform = self.data_augmentator(waveform, self.sample_rate)

        # torchaudio.load returns tensor, sample_rate
        # tensor is a Tensor with shape [channels, time]
        # we use squeeze to get ride of channels, that should be mono
        # librosa has an option to force to mono, torchaudio does not
        waveform = waveform.squeeze(0)

        if self.random_crop_secs > 0:
            
            # We make padding to allow cropping longer segments
            # (If not, we can only crop at most the duration of the shortest audio)
            if self.random_crop_samples > waveform.size(-1):
                waveform = self.pad_waveform(waveform, self.parameters.padding_type, self.random_crop_samples)
            
            # TODO torchaudio.load has frame_offset and num_frames params. Providing num_frames and frame_offset arguments is more efficient
            waveform = self.sample_audio_window(
                waveform, 
                random_crop_samples = self.random_crop_samples,
                )
        else:
            # HACK don't understand why, I have to do this slicing (which sample_audio_window does) to make dataloader work
            waveform =  waveform[:]

        waveform = self.normalize(waveform)
        
        return waveform

    
    def get_transcription(self, waveform):

        transcription = self.transcriptor.transcript(waveform)
        
        return transcription
    

    def get_transcription_tokens(self, transcription):

        indexed_tokens = self.tokenizer.encode(transcription, add_special_tokens=True)
        #tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = torch.tensor(indexed_tokens)

        return tokens_tensor


    def __getitem__(self, index):

        '''Generates one sample of data'''
        # Mandatory torch method

        # Each utterance_path is like: audio_path\tlabel
        utterance_tuple = self.utterances_paths[index].strip().split('\t')

        utterance_path = utterance_tuple[0]
        utterance_label = utterance_tuple[1]

        # By default, the resulting tensor object has dtype=torch.float32 and its value range is normalized within [-1.0, 1.0]!
        waveform, original_sample_rate = torchaudio.load(utterance_path)

        # We transcribe before processing the waveform
        if self.parameters.text_feature_extractor != 'NoneTextExtractor':
            
            # HACK
            # A real ASR should use the waveform: transcription = self.get_transcription(waveform)
            # We are using a dummy ASR that just loads transcriptions from files
            transcription = self.get_transcription(utterance_path)

            transcription_tokens = self.get_transcription_tokens(transcription)

        waveform = self.process_waveform(waveform, original_sample_rate)
        
        labels = np.array(int(utterance_label))

        if self.parameters.text_feature_extractor != 'NoneTextExtractor':
            return waveform, labels, transcription_tokens
        else:
            return waveform, labels
    

    def __len__(self):
        
        # Mandatory torch method

        return self.num_samples