import torch
from torch import nn
    

class NoneAdapter(torch.nn.Module):

    def __init__(self):
        super().__init__()
    

    def forward(self, input_tensor):

        return input_tensor
    

class LinearAdapter(torch.nn.Module):

    def __init__(self, input_vectors_dimension, output_vectors_dimension):
        super().__init__()

        self.input_vectors_dimension = input_vectors_dimension
        self.output_vectors_dimension = output_vectors_dimension
        self.adapter_layer =  nn.Sequential(
            nn.Linear(self.input_vectors_dimension, self.output_vectors_dimension),
            nn.LayerNorm(self.output_vectors_dimension),
        )
    
    
    def forward(self, input_tensor):

        output_tensor = self.adapter_layer(input_tensor)

        return output_tensor
    

class NonLinearAdapter(torch.nn.Module):

    def __init__(self, input_vectors_dimension, output_vectors_dimension):
        super().__init__()

        self.input_vectors_dimension = input_vectors_dimension
        self.output_vectors_dimension = output_vectors_dimension
        self.adapter_layer =  nn.Sequential(
            nn.Linear(self.input_vectors_dimension, self.output_vectors_dimension),
            nn.LayerNorm(self.output_vectors_dimension),
            nn.ReLU(),
        )
    
    
    def forward(self, input_tensor):

        output_tensor = self.adapter_layer(input_tensor)

        return output_tensor