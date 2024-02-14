import logging
import torchaudio
#from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
import torch
from torch import nn

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

    def __init__(self):
        super().__init__()

        self.init_extractor()
    
    
    def init_extractor(self):

        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')


    def extract_features(self, transcription_tokens_padded, transcription_tokens_mask):
  
        with torch.no_grad():

            output = self.model(transcription_tokens_padded, transcription_tokens_mask)

            # HACK we can obtain the pooled vector directly
            return output.pooler_output
            #return output.last_hidden_state

    
    def __call__(self, transcription_tokens_padded, transcription_tokens_mask):

        features = self.extract_features(transcription_tokens_padded, transcription_tokens_mask)

        return features