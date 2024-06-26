import logging
import torchaudio
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

class SpectrogramExtractor(nn.Module):

    def __init__(self, input_parameters):
        super().__init__()

        self.init_feature_extractor(input_parameters)
        

    def init_feature_extractor(self, params):

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
    

    def extract_features(self, waveform):

        features = self.feature_extractor(waveform)

        # HACK It seems that the feature extractor's output spectrogram has mel bands as rows
        features = features.transpose(1, 2)

        return features


    def __call__(self, waveform):

        logger.debug(f"waveform.size(): {waveform.size()}")

        features = self.extract_features(waveform)
        logger.debug(f"features.size(): {features.size()}")

        return features


class WavLMExtractor(nn.Module):

    def __init__(self, input_parameters):
        super().__init__()

        self.wavlm_flavor = input_parameters.wavlm_flavor
        self.init_feature_extractor()
        self.init_layers_weights()


    def init_feature_extractor(self):

        if self.wavlm_flavor == "WAVLM_BASE":
            bundle = torchaudio.pipelines.WAVLM_BASE
            self.num_layers = 12 # Layers of the Transformer of the WavLM model
            # every layer has features with 768 dimension
        elif self.wavlm_flavor == "WAVLM_BASE_PLUS":
            bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
            self.num_layers = 12 # Layers of the Transformer of the WavLM model
            # every layer has features with 768 dimension
        elif self.wavlm_flavor == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.wavlm_flavor == "WAV2VEC2_LARGE_LV60K":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.wavlm_flavor == "WAV2VEC2_XLSR_300M":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.wavlm_flavor == "WAV2VEC2_XLSR_1B":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
            self.num_layers = 48 # Layers of the Transformer of the WavLM model
            # every layer has features with 1280 dimension
        elif self.wavlm_flavor == "HUBERT_LARGE":
            bundle = torchaudio.pipelines.HUBERT_LARGE
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        else:
            raise Exception('No wavlm_flavor choice found.') 

        self.feature_extractor = bundle.get_model()

    
    def init_layers_weights(self):

        self.layer_weights = nn.Parameter(nn.functional.softmax((torch.ones(self.num_layers) / self.num_layers), dim=-1))
        

    def extract_features(self, waveform):
            
        features, _ = self.feature_extractor.extract_features(waveform)
        # level_features dims: (#B, #num_vectors, #dim_vectors = )
        
        hidden_states = torch.stack(features, dim=1)
        averaged_hidden_states = (hidden_states * self.layer_weights.view(-1, 1, 1)).sum(dim=1)

        # HACK to get only the last layer
        #averaged_hidden_states = features[-1]

        return averaged_hidden_states


    def __call__(self, waveform):

        logger.debug(f"waveform.size(): {waveform.size()}")

        features = self.extract_features(waveform)
        logger.debug(f"features.size(): {features.size()}")

        return features