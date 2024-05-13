import logging
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

class ClassifierLayer(nn.Module):

    def __init__(self, input_parameters, input_vectors_dimension):
        super().__init__()

        self.parameters = input_parameters
        self.input_vectors_dimension = input_vectors_dimension
        self.classifier_hidden_layers_width = self.parameters.classifier_hidden_layers_width
        self.classifier_hidden_layers = self.parameters.classifier_hidden_layers
        self.classifier_layer_drop_out = self.parameters.classifier_layer_drop_out

        self.init_layers()

    
    def init_layers(self):

        self.input_dropout = nn.Dropout(self.parameters.classifier_layer_drop_out)

        self.fully_connected_layer = nn.Sequential()

        self.fully_connected_layer.add_module(
                "classifier_layer_input_layer",
                nn.Sequential(
                    nn.Linear(self.input_vectors_dimension, self.classifier_hidden_layers_width), 
                    nn.LayerNorm(self.classifier_hidden_layers_width), 
                    nn.GELU(), 
                    nn.Dropout(self.parameters.classifier_layer_drop_out),
                ),
        )

        for layer_num in range(self.classifier_hidden_layers):

            hidden_layer_name = f"classifier_layer_hidden_layer_{layer_num}"

            self.fully_connected_layer.add_module(
                hidden_layer_name,
                nn.Sequential(
                    nn.Linear(self.classifier_hidden_layers_width, self.classifier_hidden_layers_width), 
                    nn.LayerNorm(self.classifier_hidden_layers_width), 
                    nn.GELU(), 
                    nn.Dropout(self.parameters.classifier_layer_drop_out),
                ),
            )
        
        self.fully_connected_layer.add_module(
            "classifier_layer_output_layer",
            nn.Sequential(
                nn.Linear(self.classifier_hidden_layers_width, self.parameters.number_classes), 
            ),
        )

    
    def forward(self, input_tensor):

        output_tensor = self.input_dropout(input_tensor)

        output_tensor = self.fully_connected_layer(output_tensor)

        return output_tensor