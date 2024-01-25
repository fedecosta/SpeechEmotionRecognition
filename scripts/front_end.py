import torch
from torch import nn
import numpy as np
from collections import OrderedDict
    

# TODO check Miquels VGG implementation
class VGG(torch.nn.Module):

    def __init__(self, vgg_n_blocks, vgg_channels):
        super().__init__()

        self.vgg_n_blocks = vgg_n_blocks # number of conv blocks
        self.vgg_channels = vgg_channels # list with number of channels of each conv block

        self.generate_conv_blocks(
            vgg_n_blocks = self.vgg_n_blocks, 
            vgg_channels = self.vgg_channels,
            )
    
    
    def generate_conv_block(self, start_block_channels, end_block_channels):

        # Create one convolutional block
        
        conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels = start_block_channels, 
                out_channels = end_block_channels, 
                kernel_size = 3, 
                stride = 1, 
                padding = 1,
                ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = end_block_channels, 
                out_channels = end_block_channels, 
                kernel_size = 3, 
                stride = 1, 
                padding = 1,
                ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2, 
                stride = 2, 
                padding = 0, 
                ceil_mode = True,
                )
            )

        return conv_block
    
    
    def generate_conv_blocks(self, vgg_n_blocks, vgg_channels):

        # Generate a nn list of vgg_n_blocks convolutional blocks
        
        self.conv_blocks = nn.Sequential() # A Python list will fail with torch
        
        start_block_channels = 1 # The first block starts with the input spectrogram, which has 1 channel
        end_block_channels = vgg_channels[0] # The first block ends with vgg_channels[0] channels

        for num_block in range(1, vgg_n_blocks + 1):
            
            conv_block_name = f"convolutional_block_{num_block}"
            conv_block = self.generate_conv_block(
                start_block_channels = start_block_channels, 
                end_block_channels = end_block_channels,
                )
            
            self.conv_blocks.add_module(conv_block_name, conv_block)

            # Update start_block_channels and end_block_channels for the next block
            if num_block < vgg_n_blocks: # If num_block = vgg_n_blocks, start_block_channels and end_block_channels must not get updated
                start_block_channels = end_block_channels # The next block will start with end_block_channels channels
                end_block_channels = vgg_channels[num_block] 
        
        # VGG ends with the end_block_channels of the last block
        self.vgg_end_channels = end_block_channels

    
    def forward(self, input_tensor):

        # input_tensor dimensions are:
        # input_tensor.size(0) = number of batches
        # input_tensor.size(1) = number of input vectors
        # input_tensor.size(2) = input vectors dimension 

        # We need to add a new dimension corresponding to the channels
        # This channel dimension will be 1 because the spectrogram has only 1 channel
        input_tensor =  input_tensor.view( 
            input_tensor.size(0),  
            input_tensor.size(1), 
            1, 
            input_tensor.size(2),
            )
            
        # We need to put the channel dimension first because nn.Conv2d need it that way
        input_tensor = input_tensor.transpose(1, 2)

        # Pass the tensor through the convolutional blocks 
        encoded_tensor = self.conv_blocks(input_tensor)
        
        # We want to flatten the output
        # For each batch, we will have encoded_tensor.size(1) hidden state vectors \
        # of size encoded_tensor.size(2) * encoded_tensor.size(3)
        output_tensor = encoded_tensor.transpose(1, 2)

        output_tensor = output_tensor.contiguous().view(
            output_tensor.size(0), 
            output_tensor.size(1), 
            output_tensor.size(2) * output_tensor.size(3)
            )

        return output_tensor


    # Method used at model.py
    def get_output_vectors_dimension(self, input_dimension):

        # Compute the front-end output's vectors dimension
        # The front-end inputs a (num_vectors, vectors_dim) tensor
        # and outputs (num_vectors / (2 ^ vgg_n_blocks)) vectors of size (vectors_dim / (2 ^ vgg_n_blocks)) * vgg_end_channels

        # Each convolutional block reduces dimension by /2
        output_vectors_dimension = input_dimension
        for num_block in range(self.vgg_n_blocks):
            output_vectors_dimension = np.ceil(np.array(output_vectors_dimension, dtype = np.float32) / 2)

        output_vectors_dimension = int(output_vectors_dimension) * self.vgg_end_channels

        return output_vectors_dimension
        

# TODO understand better the resnet and its implementation
class ResnetBlock(torch.nn.Module):

    def __init__(self, kernel_size_input, kernel_size):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            kernel_size_input, kernel_size, 3, stride=1, padding=1
        )
        self.bn1 = torch.nn.BatchNorm2d(kernel_size)
        self.conv2 = torch.nn.Conv2d(kernel_size, kernel_size, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(kernel_size)
        self.gelu = torch.nn.GELU()
        self.residual_condition = True if kernel_size_input == kernel_size else False

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_condition:
            out += x
        out = self.gelu(out)
        return out


class Resnet(torch.nn.Module):

    def __init__(self, kernel_size, group_blocks):
        super().__init__()

        self.group_blocks = group_blocks
        self.group_kernels = [
            kernel_size // 8,
            kernel_size // 4,
            kernel_size // 2,
            kernel_size,
        ]
        self.__init_blocks()

    def __init_resnet_group(self, input_kernel_size, kernel_size, num_blocks):
        layers = OrderedDict()
        layers["block0"] = ResnetBlock(input_kernel_size, kernel_size)
        for i in range(num_blocks - 1):
            layers["block" + str(i + 1)] = ResnetBlock(kernel_size, kernel_size)
        return torch.nn.Sequential(layers)

    def __init_blocks(self):
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.block1 = self.__init_resnet_group(
            1, self.group_kernels[0], self.group_blocks[0]
        )
        self.block2 = self.__init_resnet_group(
            self.group_kernels[0], self.group_kernels[1], self.group_blocks[1]
        )
        self.block3 = self.__init_resnet_group(
            self.group_kernels[1], self.group_kernels[2], self.group_blocks[2]
        )
        self.block4 = self.__init_resnet_group(
            self.group_kernels[2], self.group_kernels[3], self.group_blocks[3]
        )

    def forward(self, paddedInputTensor):
        paddedInputTensor = paddedInputTensor.view(
            paddedInputTensor.size(0),
            paddedInputTensor.size(1),
            1,
            paddedInputTensor.size(2),
        ).transpose(1, 2)

        block1_output = self.maxpool(self.block1(paddedInputTensor))
        block2_output = self.maxpool(self.block2(block1_output))
        block3_output = self.maxpool(self.block3(block2_output))
        block4_output = self.maxpool(self.block4(block3_output))

        outputTensor = block4_output.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(
            outputTensor.size(0),
            outputTensor.size(1),
            outputTensor.size(2) * outputTensor.size(3),
        )

        return outputTensor
    

    # Method used at model.py
    def get_output_vectors_dimension(self, input_dimension, output_channel = 128):

        # Compute the front-end output's vectors dimension
        
        output_dimension = np.ceil(np.array(input_dimension, dtype=np.float32) / 2)
        output_dimension = np.ceil(np.array(output_dimension, dtype=np.float32) / 2)
        output_dimension = np.ceil(np.array(output_dimension, dtype=np.float32) / 2)
        output_dimension = np.ceil(np.array(output_dimension, dtype=np.float32) / 2)
        
        output_vectors_dimension =  int(output_dimension) * output_channel

        return output_vectors_dimension
        

class Resnet34(Resnet):

    def __init__(self, kernel_size):

        super(Resnet34, self).__init__(kernel_size, [3, 4, 6, 3])


class Resnet101(Resnet):

    def __init__(self, kernel_size):

        super(Resnet101, self).__init__(kernel_size, [3, 4, 23, 3])