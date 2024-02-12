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

    def __init__(self, input_parameters, input_vectors_dimension, layer_width = 512):
        super().__init__()

        self.parameters = input_parameters
        self.input_vectors_dimension = input_vectors_dimension
        self.layer_width = layer_width

        self.init_layers()

    
    def init_layers(self):

        self.drop_out_1 = nn.Dropout(self.parameters.classifier_layer_drop_out)
        self.linear_layer_1 = nn.Linear(self.input_vectors_dimension, self.layer_width)
        self.norm_layer_1 = nn.LayerNorm(self.layer_width)
        self.act_layer_1 = nn.GELU()

        self.drop_out_2 = nn.Dropout(self.parameters.classifier_layer_drop_out)
        self.linear_layer_2 = nn.Linear(self.layer_width, self.layer_width)
        self.norm_layer_2 = nn.LayerNorm(self.layer_width)
        self.act_layer_2 = nn.GELU()

        self.drop_out_3 = nn.Dropout(self.parameters.classifier_layer_drop_out)
        self.linear_layer_3 = nn.Linear(self.layer_width, self.layer_width)
        self.norm_layer_3 = nn.LayerNorm(self.layer_width)
        self.act_layer_3 = nn.GELU()

        self.drop_out_4 = nn.Dropout(self.parameters.classifier_layer_drop_out)
        self.linear_layer_4 = nn.Linear(self.layer_width, self.layer_width)
        self.norm_layer_4 = nn.LayerNorm(self.layer_width)
        self.act_layer_4 = nn.GELU()

        self.drop_out_5 = nn.Dropout(self.parameters.classifier_layer_drop_out)
        self.linear_layer_5 = nn.Linear(self.layer_width, self.layer_width)
        self.norm_layer_5 = nn.LayerNorm(self.layer_width)
        self.act_layer_5 = nn.GELU()

        self.drop_out_6 = nn.Dropout(self.parameters.classifier_layer_drop_out)
        self.linear_layer_6 = nn.Linear(self.layer_width, self.parameters.number_classes)

    
    def forward(self, input_tensor):

        output_tensor = self.drop_out_1(input_tensor)
        output_tensor = self.linear_layer_1(output_tensor)
        output_tensor = self.norm_layer_1(output_tensor)
        output_tensor = self.act_layer_1(output_tensor)

        output_tensor = self.drop_out_2(output_tensor)
        output_tensor = self.linear_layer_2(output_tensor)
        output_tensor = self.norm_layer_2(output_tensor)
        output_tensor = self.act_layer_2(output_tensor)

        output_tensor = self.drop_out_3(output_tensor)
        output_tensor = self.linear_layer_3(output_tensor)
        output_tensor = self.norm_layer_3(output_tensor)
        output_tensor = self.act_layer_3(output_tensor)

        output_tensor = self.drop_out_4(output_tensor)
        output_tensor = self.linear_layer_4(output_tensor)
        output_tensor = self.norm_layer_4(output_tensor)
        output_tensor = self.act_layer_4(output_tensor)

        output_tensor = self.drop_out_5(output_tensor)
        output_tensor = self.linear_layer_5(output_tensor)
        output_tensor = self.norm_layer_5(output_tensor)
        output_tensor = self.act_layer_5(output_tensor)

        output_tensor = self.drop_out_6(output_tensor)
        output_tensor = self.linear_layer_6(output_tensor)

        return output_tensor