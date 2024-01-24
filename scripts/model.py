from torch import nn
from front_end import VGG
from poolings import StatisticalPooling

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
        self.init_front_end(parameters)   
        self.init_pooling_layers(parameters, device)
        self.init_adapter(parameters)
        
        #print(f"self.front_end_output_vectors_dimension: {self.front_end_output_vectors_dimension}")

        if False:
            
            self.__initFullyConnectedBlock(parameters)
            
            self.am_softmax_layer = AMSoftmax(
                parameters.embedding_size, 
                parameters.number_speakers, 
                s = parameters.scaling_factor, 
                m = parameters.margin_factor, 
                )
    

    def init_front_end(self, parameters):

        if parameters.front_end == 'VGG':

            # Set the front-end component that will take the spectrogram and generate complex features
            self.front_end = VGG(parameters.vgg_n_blocks, parameters.vgg_channels)
                
            # Calculate the size of the hidden state vectors (output of the front-end)
            self.front_end_output_vectors_dimension = self.front_end.get_output_vectors_dimension(
                parameters.front_end_input_vectors_dim, 
                )
 

    def init_pooling_layers(self, parameters, device):    

        # Set the pooling component that will take the front-end features and summarize them in a context vector

        self.pooling_method = parameters.pooling_method

        # HACK until a real pooling is developed
        self.pooling_layer = StatisticalPooling(
            emb_in = self.front_end_output_vectors_dimension,
        )

        if False:
            # New Pooling classes
            if self.pooling_method == 'SelfAttentionAttentionPooling':
                self.poolingLayer = SelfAttentionAttentionPooling(
                    emb_in = self.hidden_states_dimension,
                    emb_out = parameters.pooling_output_size,
                    positional_encoding = parameters.pooling_positional_encoding,
                    device = device,
                    )
            elif self.pooling_method == 'MultiHeadAttentionAttentionPooling':
                self.poolingLayer = MultiHeadAttentionAttentionPooling(
                    emb_in = self.hidden_states_dimension,
                    emb_out = parameters.pooling_output_size,
                    heads = parameters.pooling_heads_number,
                    positional_encoding = parameters.pooling_positional_encoding,
                    device = device,
                    )
            elif self.pooling_method == 'TransformerStackedAttentionPooling':
                self.poolingLayer = TransformerStackedAttentionPooling(
                    emb_in = self.hidden_states_dimension,
                    emb_out = parameters.pooling_output_size,
                    n_blocks = parameters.transformer_n_blocks, 
                    expansion_coef = parameters.transformer_expansion_coef, 
                    attention_type = parameters.transformer_attention_type, 
                    drop_out_p = parameters.transformer_drop_out, 
                    heads = parameters.pooling_heads_number,
                    positional_encoding = parameters.pooling_positional_encoding,
                    device = device,
                    )
    

    def init_adapter(self, parameters):

        self.adapter_layer = nn.Linear(parameters.pooling_output_size, parameters.number_classes)

    
    def forward(self, input_tensor, label = None):

        # Mandatory torch method
        # Set the net's forward pass

        #print(f"input_tensor.size(): {input_tensor.size()}")

        encoder_output = self.front_end(input_tensor)
        #print(f"encoder_output.size(): {encoder_output.size()}")

        pooling_output = self.pooling_layer(encoder_output)
        #print(f"pooling_output.size(): {pooling_output.size()}")

        # adapter_output are logits, softmax will be applied within the loss
        adapter_output = self.adapter_layer(pooling_output)
        #print(f"adapter_output.size(): {adapter_output.size()}")
    
        return adapter_output
    
    
    
    
    if False:

        


        def __initFullyConnectedBlock(self, parameters):

            # Set the set of fully connected layers that will take the pooling context vector

            # TODO abstract the FC component in a class with a forward method like the other components
            # TODO Get also de RELUs in this class
            # Should we batch norm and relu the last layer?

            if self.pooling_method in ('SelfAttentionAttentionPooling', 'MultiHeadAttentionAttentionPooling', 'TransformerStackedAttentionPooling'):
                # New Pooling classes output size is different from old poolings
                self.fc1 = nn.Linear(parameters.pooling_output_size, parameters.embedding_size)
            else:
                self.fc1 = nn.Linear(self.hidden_states_dimension, parameters.embedding_size)
            self.b1 = nn.BatchNorm1d(parameters.embedding_size)
            self.fc2 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
            self.b2 = nn.BatchNorm1d(parameters.embedding_size)
            self.fc3 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
            self.b3 = nn.BatchNorm1d(parameters.embedding_size)

            self.drop_out = nn.Dropout(parameters.bottleneck_drop_out)
            self.softmax = nn.Softmax(dim=1)


        


        # This method is used at test (or valid) time
        def get_embedding(self, input_tensor):

            # TODO should we use relu and bn in every layer?d

            encoder_output = self.front_end(input_tensor)

            # TODO seems that alignment is not used anywhere
            embedding_0, alignment = self.poolingLayer(encoder_output)

            # TODO should we use relu and bn in every layer?

            # NO DROPOUT HERE
            embedding_1 = self.fc1(embedding_0)
            embedding_1 = F.relu(embedding_1)
            embedding_1 = self.b1(embedding_1)

            embedding_2 = self.fc2(embedding_1)
            embedding_2 = F.relu(embedding_2)
            embedding_2 = self.b2(embedding_2)
        
            return embedding_2 

