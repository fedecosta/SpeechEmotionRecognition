from torch import nn
import torch
from torch.nn import functional as F
import logging
import copy
import math

# Based on https://peterbloem.nl/blog/transformers
# TODO make dim asserts in every new class

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

# 1 - Sequence to sequence components (sequence to sequence blocks, the input dimension is the same than the output dimension)

class NoneSeqToSeq(torch.nn.Module):

    def __init__(self):
        super().__init__()
    

    def forward(self, input_tensor):

        return input_tensor
    

class SelfAttention(nn.Module):

    """
    Sequence to sequence component, the input dimension is the same than the output dimension.
    Sequence length is not fixed.
    Self-attention without trainable parameters.
    """

    def __init__(self):

        super().__init__()


    def forward(self, input_tensors):

        #print(f"input_tensors.size(): {input_tensors.size()}")

        raw_weights = torch.bmm(input_tensors, input_tensors.transpose(1, 2))

        # TODO If we want to analyze the attention weights, we should analyze weights
        weights = F.softmax(raw_weights, dim = 2)

        output = torch.bmm(weights, input_tensors)

        return output
    

class MultiHeadAttention(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        emb_in is the dimension of every input vector (embedding).
        heads is the number of heads to use in the Multi-Head Attention.
    """

    def __init__(self, emb_in, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # HACK we force the same input and output dimension
        self.heads = heads

        self.init_matrix_transformations()
    

    def init_matrix_transformations(self):

        # Matrix transformations to stack every head keys, queries and values matrices
        self.to_keys = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_queries = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)
        self.to_values = nn.Linear(self.emb_in, self.emb_out * self.heads, bias=False)

        # Linear projection. For each input vector we get self.heads heads, we project them into only one.
        self.unify_heads = nn.Linear(self.heads * self.emb_out, self.emb_out)
    
    
    def forward(self, input_tensors):

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        keys = self.to_keys(input_tensors).view(b, t, self.heads, self.emb_out)
        queries = self.to_queries(input_tensors).view(b, t, self.heads, self.emb_out)
        values = self.to_values(input_tensors).view(b, t, self.heads, self.emb_out)

        # 1 - Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)

        # - Instead of dividing the dot products by sqrt(e), we scale the queries and keys.
        #   This should be more memory efficient
        queries = queries / (self.emb_out ** (1/4))
        keys    = keys / (self.emb_out ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * self.heads, t, t), f'Matrix has size {dot.size()}, expected {(b * self.heads, t, t)}.'

        dot = F.softmax(dot, dim = 2) # dot now has row-wise self-attention probabilities

        # 2 - Apply the self attention to the values
        output = torch.bmm(dot, values).view(b, self.heads, t, self.emb_out)

        # swap h, t back
        output = output.transpose(1, 2).contiguous().view(b, t, self.heads * self.emb_out)

        # unify heads
        output = self.unify_heads(output)

        return output


class TransformerBlock(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        One Transformer block.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """

    def __init__(self, emb_in, expansion_coef, drop_out_p, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we want the same dimension
        self.expansion_coef = expansion_coef
        self.drop_out_p = drop_out_p
        self.heads = heads
        

        self.init_attention_layer()
        self.init_norm_layers()
        self.init_feed_forward_layer()
        self.drop_out = nn.Dropout(drop_out_p)


    def init_attention_layer(self):

        self.attention_layer = MultiHeadAttention(self.emb_in, self.heads)


    def init_norm_layers(self):

        self.norm1 = nn.LayerNorm(self.emb_out)
        self.norm2 = nn.LayerNorm(self.emb_out)


    def init_feed_forward_layer(self):

        self.feed_forward_layer = nn.Sequential(
            nn.Linear(self.emb_out, self.expansion_coef * self.emb_out),
            nn.ReLU(),
            nn.Linear(self.expansion_coef * self.emb_out, self.emb_out),
            )


    def forward(self, input_tensors):

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # Pass through the attention component
        attention_layer_output = self.attention_layer(input_tensors)

        # Make the skip connection
        skip_connection_1 = attention_layer_output + input_tensors

        # Normalization layer
        normalized_1 = self.norm1(skip_connection_1)

        # Feed forward component
        feed_forward = self.feed_forward_layer(self.drop_out(normalized_1))
        
        # Make the skip connection
        skip_connection_2 = feed_forward + normalized_1

        # Normalization layer
        norm_attended_2 = self.norm2(skip_connection_2)

        # Output
        output = norm_attended_2

        return output


class TransformerStacked(nn.Module):

    """
        Sequence to sequence component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Stack of n_blocks Transformer blocks.
        emb_in is the dimension of every input vector (embedding).
        expansion_coef is the number you want to multiply the size of the hidden layer of the feed forward net.
        attention_type is the type of attention to use in the attention component.
        heads is the number of heads to use in the attention component, if Multi-Head Attention is used.
    """

    def __init__(self, emb_in, n_blocks, expansion_coef, drop_out_p, heads):

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_in # we force the same input and output dimension
        self.n_blocks = n_blocks
        self.expansion_coef = expansion_coef
        self.drop_out_p = drop_out_p
        self.heads = heads

        self.init_transformer_blocks()


    def init_transformer_block(self, emb_in, expansion_coef, drop_out_p, heads):

        # Init one transformer block

        transformer_block = TransformerBlock(emb_in, expansion_coef, drop_out_p, heads)

        return transformer_block


    def init_transformer_blocks(self):

        self.transformer_blocks = nn.Sequential()

        for num_block in range(self.n_blocks):

            transformer_block_name = f"transformer_block_{num_block}"
            transformer_block = self.init_transformer_block(self.emb_in, self.expansion_coef, self.drop_out_p, self.heads)
                
            self.transformer_blocks.add_module(transformer_block_name, transformer_block)


    def forward(self, input_tensors):

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        transformer_output = self.transformer_blocks(input_tensors)

        output = transformer_output

        return output


# We call "Reduced Multi-Head Attention" to the implementation of the paper: https://arxiv.org/abs/2007.13199

def new_parameter(*size):

    out = torch.nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)

    return out


def innerKeyValueAttention(query, key, value):

    d_k = query.size(-1)
    scores = torch.diagonal(torch.matmul(key, query) / math.sqrt(d_k), dim1=-2, dim2=-1).view(value.size(0),value.size(1), value.size(2))
    p_attn = F.softmax(scores, dim = -2)
    weighted_vector = value * p_attn.unsqueeze(-1)
    ct = torch.sum(weighted_vector, dim=1)
    return ct, p_attn


class ReducedMultiHeadAttention(nn.Module):
    
    def __init__(self, encoder_size, heads_number):
        super().__init__()

        self.encoder_size = encoder_size
        assert self.encoder_size % heads_number == 0 # d_model
        self.head_size = self.encoder_size // heads_number 
        self.heads_number = heads_number
        self.query = new_parameter(self.head_size, self.heads_number)
        self.aligmment = None

        
    def getAlignments(self,ht):

        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)

        return self.alignment 
    

    def getHeadsContextVectors(self,ht):    

        batch_size = ht.size(0)
        logger.debug(f"ht.size(): {ht.size()}")
        logger.debug(f"batch_size: {batch_size}")
        logger.debug(f"self.head_size: {self.head_size}")
        logger.debug(f"self.heads_number: {self.heads_number}")
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)

        return headsContextVectors


    def forward(self, ht):

        logger.debug(f"ht.size(): {ht.size()}")

        headsContextVectors = self.getHeadsContextVectors(ht)
        logger.debug(f"headsContextVectors.size(): {headsContextVectors.size()}")

        # original line
        #return headsContextVectors.view(headsContextVectors.size(0),-1), copy.copy(self.alignment)
        
        return headsContextVectors


# ---------------------------------------------------------------------
# 2 - Pooling components (sequence to one components, the input dimension is the same than the output dimension)

class StatisticalPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their average as output.
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()
        
        self.emb_in = emb_in 


    def forward(self, input_tensors):

        logger.debug(f"input_tensors.size(): {input_tensors.size()}")

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        # Get the average of the input vectors (dim = 0 is the batch dimension)
        output = input_tensors.mean(dim = 1)

        return output


class AttentionPooling(nn.Module):

    """
        Sequence to one component, the input dimension is the same than the output dimension.
        Sequence length is not fixed.
        Given n vectors, takes their weighted average as output. These weights comes from an attention mechanism.
        It can be seen as a One Head Self-Attention, where a unique query is used and input vectors are the values and keys.   
        emb_in is the dimension of every input vector (embedding).
    """

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in
        self.init_query()

        
    def init_query(self):

        # Init the unique trainable query.
        self.query = torch.nn.Parameter(torch.FloatTensor(self.emb_in, 1))
        torch.nn.init.xavier_normal_(self.query)


    def forward(self, input_tensors):

        #logger.debug(f"input_tensors.size(): {input_tensors.size()}")

        #logger.debug(f"self.query[0]: {self.query[0]}")

        b, t, e = input_tensors.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        attention_scores = torch.matmul(input_tensors, self.query)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        #logger.debug(f"self.query.size(): {self.query.size()}")
        attention_scores = attention_scores.squeeze(dim = -1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        attention_scores = F.softmax(attention_scores, dim = 1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")
        attention_scores = attention_scores.unsqueeze(dim = -1)
        #logger.debug(f"attention_scores.size(): {attention_scores.size()}")

        output = torch.bmm(attention_scores.transpose(1, 2), input_tensors)
        #logger.debug(f"output.size(): {output.size()}")
        output = output.view(output.size()[0], output.size()[1] * output.size()[2])
        #logger.debug(f"output.size(): {output.size()}")
        
        return output
