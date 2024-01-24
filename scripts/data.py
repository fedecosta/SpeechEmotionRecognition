from torch.utils import data
import numpy as np
import torchaudio
import torch
from random import randint
import copy

class TrainDataset(data.Dataset):

    def __init__(self, utterances_paths, input_parameters, random_crop_secs, sample_rate = 16000):
        
        self.utterances_paths = utterances_paths
        self.parameters = copy.deepcopy(input_parameters) # I suspect when instantiating two datasets the parameters are overrided
        self.sample_rate = sample_rate
        self.random_crop_secs = random_crop_secs
        self.random_crop_samples = int(self.random_crop_secs * self.sample_rate)
        self.num_samples = len(utterances_paths)
        self.set_feature_extractor()

    
    def set_feature_extractor(self):

        self.spectogram_extractor = torchaudio.transforms.MelSpectrogram(
            n_fft = 512,
            win_length = int(self.sample_rate * 0.025),
            hop_length = int(self.sample_rate * 0.01),
            n_mels = self.parameters.front_end_input_vectors_dim,
            mel_scale = "slaney",
            window_fn = torch.hamming_window,
            f_max = self.sample_rate // 2,
            center = False,
            normalized = False,
            norm = "slaney",
        )

    
    def sample_audio_window(self, waveform, random_crop_samples):

        waveform_total_samples = waveform.size()[-1]
        
        # TODO maybe we can add an assert to check that random_crop_samples <= waveform_total_samples (will it slow down the process?)
        random_start_index = randint(0, waveform_total_samples - random_crop_samples)
        end_index = random_start_index + random_crop_samples
        
        cropped_waveform =  waveform[random_start_index : end_index]

        return cropped_waveform

    
    def get_feature_vector(self, utterance_path):

        waveform, sample_rate = torchaudio.load(utterance_path)

        # torchaudio.load returns tensor, sample_rate
        # tensor is a Tensor with shape [channels, time]
        # we use squeeze to get ride of channels, that should be mono
        # librosa has an option to force to mono, torchaudio does not
        waveform = waveform.squeeze(0)

        if self.random_crop_secs > 0:
            # TODO torchaudio.load has frame_offset and num_frames params
            waveform = self.sample_audio_window(
                waveform, 
                random_crop_samples = self.random_crop_samples,
                )
        else:
            # HACK don't understand why, I have to do this slicing (which sample_audio_window does) to make dataloader work
            waveform =  waveform[:]

        features = self.spectogram_extractor(waveform)

        # It seems that the feature extractor's output spectrogram has mel bands as rows
        features = features.transpose(0, 1)

        return features

    
    def __getitem__(self, index):

        '''Generates one sample of data'''
        # Mandatory torch method

        # Each utterance_path is like: audio_path\tlabel
        utterance_tuple = self.utterances_paths[index].strip().split('\t')

        utterance_path = utterance_tuple[0]
        utterance_label = utterance_tuple[1]

        features = self.get_feature_vector(utterance_path)
        labels = np.array(int(utterance_label))
        
        return features, labels
    

    def __len__(self):
        
        # Mandatory torch method

        return self.num_samples