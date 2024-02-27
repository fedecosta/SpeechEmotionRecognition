import logging
import torchaudio
#from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
import torch
from torch import nn
import os

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


class ASRDummy(nn.Module):

    # https://pytorch.org/audio/main/generated/torchaudio.pipelines.RNNTBundle.html#torchaudio.pipelines.RNNTBundle

    def __init__(self):
        super().__init__()


    def transcript(self, utterance_path):

        file_name = utterance_path.split("/")[-1].replace(".wav", ".txt")
        #transcription_path = os.path.join("/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Transcripts/Transcripts", file_name)
        #transcription_path = os.path.join("/home/usuaris/veussd/federico.costa/datasets/msp_podcast/custom_transcriptions/", file_name)
        transcription_path = os.path.join("/home/usuaris/veussd/federico.costa/datasets/msp_podcast/whisper_transcriptions/", file_name)
        
        with open(transcription_path, 'r') as data_labels_file:
            transcription = data_labels_file.readlines()
        
        # HACK
        if len(transcription) != 1: 
            logger.debug(f"utterance_path: {utterance_path}")
            transcription = "..."
        else:
            transcription = transcription[0]

        return transcription
    

class ASRModel(nn.Module):

    # https://pytorch.org/audio/main/generated/torchaudio.pipelines.RNNTBundle.html#torchaudio.pipelines.RNNTBundle

    def __init__(self):
        super().__init__()

        self.init_transcriptor()

    
    def init_transcriptor(self):

        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH

        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()

        self.sample_rate = bundle.sample_rate
        logger.info(f"ASRModel sample rate: {self.sample_rate}")


    def transcript(self, waveform):

        with torch.no_grad():

            # Produce mel-scale spectrogram features.
            logger.debug(f"waveform.size(): {waveform.size()}")

            # HACK we need to squeeze the waveform because it is needed that way: 
            # https://pytorch.org/audio/main/generated/torchaudio.pipelines.RNNTBundle.html#torchaudio.pipelines.RNNTBundle
            # Search for the line 'length = torch.tensor([features.shape[0]])' in https://pytorch.org/audio/0.11.0/_modules/torchaudio/pipelines/rnnt_pipeline.html 
            features, length = self.feature_extractor(waveform.squeeze())

            # Generate top-10 hypotheses.
            hypotheses = self.decoder(features, length, 10)

        # For top hypothesis, convert predicted tokens to text.
        transcription = self.token_processor(hypotheses[0][0])

        return transcription
    

class TextBERTExtractor(nn.Module):

    def __init__(self, input_parameters):
        super().__init__()

        self.bert_flavor = input_parameters.bert_flavor
        self.init_extractor()
    
    
    def init_extractor(self):

        if self.bert_flavor == "BERT_BASE_UNCASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
            # model outputs features with 768 dimension
        elif self.bert_flavor == "BERT_BASE_CASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
            # model outputs features with 768 dimension
        elif self.bert_flavor == "BERT_LARGE_UNCASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-uncased')
            # model outputs features features with 1024 dimension
        elif self.bert_flavor == "BERT_LARGE_CASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-cased')
            # model outputs features features with 1024 dimension
        else:
            raise Exception('No bert_flavor choice found.')


    def extract_features(self, transcription_tokens_padded, transcription_tokens_mask):
  
        with torch.no_grad():

            output = self.model(transcription_tokens_padded, transcription_tokens_mask)
            
            # we can obtain the pooled vector directly
            #features =  output.pooler_output
            
            # we can obtain the last layer features
            features = output.last_hidden_state
            # features dims: (#B, #num_vectors, #dim_vectors = 768)

            logger.debug(f"features.size(): {features.size()}")

        return features

    
    def __call__(self, transcription_tokens_padded, transcription_tokens_mask):

        features = self.extract_features(transcription_tokens_padded, transcription_tokens_mask)

        return features