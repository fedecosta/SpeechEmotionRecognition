import logging
from torch import nn
import torch
from feature_extractor import SpectrogramExtractor, WavLMExtractor
from text_feature_extractor import TextBERTExtractor
from front_end import VGG, Resnet34, Resnet101, NoneFrontEnd
from adapter import NoneAdapter, LinearAdapter, NonLinearAdapter
from poolings import NoneSeqToSeq, SelfAttention, MultiHeadAttention, TransformerStacked, ReducedMultiHeadAttention
from poolings import StatisticalPooling, AttentionPooling
from classifier_layer import ClassifierLayer

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

if False:
    import torch
    
    from torch.nn import functional as F
    from poolings_original import Attention, MultiHeadAttention, DoubleMHA
    from poolings import SelfAttentionAttentionPooling, MultiHeadAttentionAttentionPooling, TransformerStackedAttentionPooling
    
    from loss import AMSoftmax

class Classifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
     
        self.device = device
        self.init_feature_extractor(parameters)
        self.init_text_feature_extractor(parameters)
        self.init_front_end(parameters)   
        self.init_adapter_layer(parameters)
        self.init_pooling_component(parameters)
        self.init_classifier_layer(parameters)
    

    def init_feature_extractor(self, parameters):

        if parameters.feature_extractor == 'SpectrogramExtractor':
            self.feature_extractor = SpectrogramExtractor(parameters)
        elif parameters.feature_extractor == 'WavLMExtractor':
            self.feature_extractor = WavLMExtractor(parameters)
        else:
            raise Exception('No Feature Extractor choice found.') 
        
        for name, parameter in self.feature_extractor.named_parameters():
            
            # Freeze all wavLM parameters except layers weights and the last layer
            #if name != "layer_weights" and "transformer.layers.11" not in name:
            
            # Freeze all wavLM parameters except layers weights
            if name != "layer_weights":
                logger.info(f"Setting {name} to requires_grad = False")
                parameter.requires_grad = False

    
    def init_text_feature_extractor(self, parameters):

        if parameters.text_feature_extractor == 'TextBERTExtractor':
            self.text_feature_extractor = TextBERTExtractor()

            for name, parameter in self.text_feature_extractor.named_parameters():
                # TODO allow to train some parameters
                # Freeze all BERT parameters except layers weights and the last layer
                #if name != "layer_weights" and "transformer.layers.11" not in name:
                # Freeze all BERT parameters except layers weights
                if True:#name != "layer_weights":
                    logger.info(f"Setting {name} to requires_grad = False")
                    parameter.requires_grad = False
        else:
            self.text_feature_extractor = None
            logger.info('No Text Feature Extractor selected.')
    
    
    def init_front_end(self, parameters):

        if parameters.front_end == 'VGG':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = VGG(parameters.vgg_n_blocks, parameters.vgg_channels)
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                )

        elif parameters.front_end == 'Resnet34':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = Resnet34(
                256, # HACK set as parameter?
                )
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                256, # HACK set as parameter?
                )
        
        elif parameters.front_end == 'Resnet101':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = Resnet101(256)
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                256,
                )

        elif parameters.front_end == 'NoneFrontEnd':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = NoneFrontEnd()
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.feature_extractor_output_vectors_dimension, 
                )
        
        else:
            raise Exception('No Front End choice found.')  
        

    def init_adapter_layer(self, parameters):
        NoneAdapter, LinearAdapter, NonLinearAdapter

        if parameters.adapter == 'NoneAdapter':
            self.adapter_layer = NoneAdapter()
            parameters.adapter_output_vectors_dimension = self.front_end_output_vectors_dimension

        elif parameters.adapter == 'LinearAdapter':
            self.adapter_layer = LinearAdapter(self.front_end_output_vectors_dimension, parameters.adapter_output_vectors_dimension)
        elif parameters.adapter == 'NonLinearAdapter':
            self.adapter_layer = NonLinearAdapter(self.front_end_output_vectors_dimension, parameters.adapter_output_vectors_dimension)
        else:
            raise Exception('No Adapter choice found.') 

    
    def init_seq_to_seq_layer(self, parameters):
        
            self.seq_to_seq_method = parameters.seq_to_seq_method
            self.seq_to_seq_input_vectors_dimension = parameters.adapter_output_vectors_dimension

            # HACK ReducedMultiHeadAttention seq to seq input and output dimensions don't match
            if self.seq_to_seq_method == 'ReducedMultiHeadAttention':
                self.seq_to_seq_output_vectors_dimension = self.seq_to_seq_input_vectors_dimension // parameters.seq_to_seq_heads_number
            else:
                self.seq_to_seq_output_vectors_dimension = self.seq_to_seq_input_vectors_dimension

            if self.seq_to_seq_method == 'NoneSeqToSeq':
                self.seq_to_seq_layer = NoneSeqToSeq()
            
            elif self.seq_to_seq_method == 'SelfAttention':
                self.seq_to_seq_layer = SelfAttention()

            elif self.seq_to_seq_method == 'MultiHeadAttention':
                self.seq_to_seq_layer = MultiHeadAttention(
                    emb_in = self.seq_to_seq_input_vectors_dimension,
                    heads = parameters.seq_to_seq_heads_number,
                )

            elif self.seq_to_seq_method == 'TransformerStacked':
                self.seq_to_seq_layer = TransformerStacked(
                    emb_in = self.seq_to_seq_input_vectors_dimension,
                    n_blocks = parameters.transformer_n_blocks,
                    expansion_coef = parameters.transformer_expansion_coef,
                    drop_out_p = parameters.transformer_drop_out,
                    heads = parameters.seq_to_seq_heads_number,
                )
            
            elif self.seq_to_seq_method == 'ReducedMultiHeadAttention':
                self.seq_to_seq_layer = ReducedMultiHeadAttention(
                    encoder_size = self.seq_to_seq_input_vectors_dimension,
                    heads_number = parameters.seq_to_seq_heads_number,
                )

            else:
                raise Exception('No Seq to Seq choice found.')  
    

    def init_seq_to_one_layer(self, parameters):

            self.seq_to_one_method = parameters.seq_to_one_method
            self.seq_to_one_input_vectors_dimension = self.seq_to_seq_output_vectors_dimension
            self.seq_to_one_output_vectors_dimension = self.seq_to_one_input_vectors_dimension

            if self.seq_to_one_method == 'StatisticalPooling':
                self.seq_to_one_layer = StatisticalPooling(
                        emb_in = self.seq_to_one_input_vectors_dimension,
                    )

            elif self.seq_to_one_method == 'AttentionPooling':
                self.seq_to_one_layer = AttentionPooling(
                    emb_in = self.seq_to_one_input_vectors_dimension,
                )
            
            else:
                raise Exception('No Seq to One choice found.') 
            
    
    def init_pooling_component(self, parameters):    

        # Set the pooling component that will take the front-end features and summarize them in a context vector
        # This component applies first a sequence to sequence layer and then a sequence to one layer.

        self.init_seq_to_seq_layer(parameters)
        self.init_seq_to_one_layer(parameters)
    

    def init_classifier_layer(self, parameters):

        if self.text_feature_extractor:
            # In this case we will concatenate text and acoustic pooled vectors
            # HACK
            self.classifier_layer_input_vectors_dimension = 2 * self.seq_to_one_output_vectors_dimension
        else:
            self.classifier_layer_input_vectors_dimension = self.seq_to_one_output_vectors_dimension
        
        logger.debug(f"self.classifier_layer_input_vectors_dimension: {self.classifier_layer_input_vectors_dimension}")
        
        self.classifier_layer = ClassifierLayer(parameters, self.classifier_layer_input_vectors_dimension)


    def forward(self, input_tensor, transcription_tokens_padded = None, transcription_tokens_mask = None):

        # Mandatory torch method
        # Set the net's forward pass

        logger.debug(f"input_tensor.size(): {input_tensor.size()}")

        # Acoustic-based components

        feature_extractor_output = self.feature_extractor(input_tensor)
        logger.debug(f"feature_extractor_output.size(): {feature_extractor_output.size()}")

        encoder_output = self.front_end(feature_extractor_output)
        logger.debug(f"encoder_output.size(): {encoder_output.size()}")

        adapter_output = self.adapter_layer(encoder_output)
        logger.debug(f"adapter_output.size(): {adapter_output.size()}")

        seq_to_seq_output = self.seq_to_seq_layer(adapter_output)
        logger.debug(f"seq_to_seq_output.size(): {seq_to_seq_output.size()}")

        seq_to_one_output = self.seq_to_one_layer(seq_to_seq_output)
        logger.debug(f"seq_to_one_output.size(): {seq_to_one_output.size()}")

        # Text-based components

        if self.text_feature_extractor:
            text_feature_extractor_output = self.text_feature_extractor(transcription_tokens_padded, transcription_tokens_mask)
            logger.debug(f"text_feature_extractor_output.size(): {text_feature_extractor_output.size()}")

        # classifier_output are logits, softmax will be applied within the loss
        if self.text_feature_extractor:
            classifier_input = torch.cat([seq_to_one_output, text_feature_extractor_output], 1)
        else:
            classifier_input = seq_to_one_output
        logger.debug(f"classifier_input.size(): {classifier_input.size()}")

        classifier_output = self.classifier_layer(classifier_input)
        logger.debug(f"classifier_output.size(): {classifier_output.size()}")
    
        return classifier_output


