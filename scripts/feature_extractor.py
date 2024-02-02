import logging
import torchaudio
import torch

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

class FeatureExtractor(torch.nn.Module):

    def __init__(self, input_parameters):
        super().__init__()
        
        self.feature_extractor_type = input_parameters.feature_extractor
        self.init_feature_extractor(input_parameters)
        

    
    def init_feature_extractor(self, params):

        if self.feature_extractor_type == 'Spectrogram':

            self.feature_extractor = torchaudio.transforms.MelSpectrogram(
                n_fft = 512,
                win_length = int(params.sample_rate * 0.025),
                hop_length = int(params.sample_rate * 0.01),
                n_mels = params.feature_extractor_output_vectors_dimension,
                mel_scale = "slaney",
                window_fn = torch.hamming_window,
                f_max = params.sample_rate // 2,
                center = False,
                normalized = False,
                norm = "slaney",
            )
        else:
            raise Exception('No Feature Extractor choice found.')
    

    def extract_features(self, waveform):

        features = self.feature_extractor(waveform)

        # HACK It seems that the feature extractor's output spectrogram has mel bands as rows
        if self.feature_extractor_type == 'Spectrogram':
            features = features.transpose(1, 2)

        return features


    def __call__(self, waveform):

        logger.debug(f"waveform.size(): {waveform.size()}")

        features = self.extract_features(waveform)
        logger.debug(f"features.size(): {features.size()}")

        return features