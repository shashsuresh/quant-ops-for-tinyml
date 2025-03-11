'''
A collection of functions that are used to build
a quantized network using the layers defined in `quantized_operations`
'''
from conversions import to_pytorch_tensor
from quantized_operations import QuantizedConv2D
from quantized_operations import QuantizedAvgPool
from quantized_operations import QuantizedElementWise
from quantized_operations import QuantizedMBBlock
import torch as tch

def construct_q_conv(conv_custom, w_bit=8, a_bit=None):
    '''
    Create a QuantizedConv2D layer from the description of a 
    convolution layer provided in a `.pkl` file
    '''
    # Arguments for our custom quantized Conv2D layer
    conv_args = {
        'zero_x': to_pytorch_tensor(conv_custom['params']['x_zero']),
        'zero_w': to_pytorch_tensor(0),
        'zero_y' : to_pytorch_tensor(conv_custom['params']['y_zero'])
    }

    # Calculate and store effective scale
    conv_args['effective_scale'] = to_pytorch_tensor(conv_custom['params']['x_scale']).double() * to_pytorch_tensor(conv_custom['params']['w_scales']).double() / to_pytorch_tensor(conv_custom['params']['y_scale']).double()

    # We want tuple kernel sizes, not ints!
    if isinstance(conv_custom['kernel_size'], int):
        conv_args['kernel_size'] = (conv_custom['kernel_size'],) * 2
    
    # Padding based on kernel sizes
    padding = ((conv_custom['kernel_size'][0] - 1) // 2, (conv_custom['kernel_size'][1] - 1) // 2)

    # Create custom quantized convolution layer from dict
    conv = QuantizedConv2D(conv_custom['in_channel'], conv_custom['out_channel'], conv_custom['kernel_size'],
                           padding=padding, stride=conv_custom['stride'], groups=conv_custom['groups'],
                           w_bit=w_bit, a_bit=a_bit, **conv_args)

    # Set and weighs and bias of the newly created quantized convolution layer from the .pkl file
    conv.weight.data = to_pytorch_tensor(conv_custom['params']['weight'])
    conv.bias.data = to_pytorch_tensor(conv_custom['params']['bias'])
    # These parameters are needed for obtaining quantized weights and biases for the last FC added when creating the model
    conv.x_scale = conv_custom['params']['x_scale']
    conv.y_scale = conv_custom['params']['y_scale']
    return conv

def construct_q_residual(res_custom, pt_wise2_custom, n_bit):
    '''
    Create a residual block from the dict in the `pkl` file
    '''
    if 'kernel_size' in res_custom:
        # Implies there is a convolution in the residual connection - typically to adjust dimensions
        
        # Scales for x
        scale_x = res_custom['params']['y_scale']
        zero_x = res_custom['params']['y_zero']

        # Scales for convolution (get from prev output)
        scale_convolution = pt_wise2_custom['params']['y_scale']
        zero_convolution = pt_wise2_custom['params']['y_zero']

        # scales for y
        scale_y = res_custom['params']['out_scale']
        zero_y = res_custom['params']['out_scale']

        # Create quantized element wise layer based on parsed scales
        q_add = QuantizedElementWise('add', to_pytorch_tensor(zero_x), to_pytorch_tensor(zero_convolution),
                                     to_pytorch_tensor(zero_y),to_pytorch_tensor(scale_x), to_pytorch_tensor(scale_convolution),
                                     to_pytorch_tensor(scale_y),
                                     )

        # usually present to correct dimensions for addition
        residual_convolution = construct_q_conv(res_custom, w_bit=n_bit)

    else:
        # No residual convolution, just add input to regular conv. output

        scale_x = res_custom['params']['x_scale']
        zero_x = res_custom['params']['x_zero']

        scale_convolution = pt_wise2_custom['params']['y_scale']
        zero_convolution = pt_wise2_custom['params']['y_zero']

        scale_y = res_custom['params']['y_scale']
        zero_y = res_custom['params']['y_zero']

        # Create quantized element wise layer based on parsed scales
        q_add = QuantizedElementWise('add', to_pytorch_tensor(zero_x), to_pytorch_tensor(zero_convolution),
                                     to_pytorch_tensor(zero_y),to_pytorch_tensor(scale_x), to_pytorch_tensor(scale_convolution),
                                     to_pytorch_tensor(scale_y),
                                     )
        
        # There is no convolution to correct dimension, so we don't add anything to this part of the layer
        residual_convolution = None

    return q_add, residual_convolution

def construct_q_block(blk_custom, n_bit=8):
    '''
    Create a quantized "block" (multiple convs) from the blocks in the "pkl" file
    '''
    block = tch.nn.Sequential()

    # SE is not implemented in the original version either!
    if blk_custom['se'] != None:
        raise NotImplemented

    # If present, add point wise conv to the block list
    if blk_custom['pointwise1'] != None:
        block.append(construct_q_conv(blk_custom['pointwise1'], w_bit=n_bit))
    
    # If present, add depth wise conv to the block list
    if blk_custom['depthwise'] != None:
        block.append(construct_q_conv(blk_custom['depthwise'], w_bit=n_bit))
    
    # If present, add point wise2 conv to the block list
    if blk_custom['pointwise2'] != None:
        block.append(construct_q_conv(blk_custom['pointwise2'], w_bit=n_bit))

    if blk_custom['residual'] != None:
        q_add, residual = construct_q_residual(blk_custom['residual'], blk_custom['pointwise2'], n_bit=n_bit)
    else:
        q_add = None
        residual = None

    return QuantizedMBBlock(block, q_add, residual_conv=residual, a_bit=n_bit)
    

def construct_quantized_torch_nn(pkl_file, n_bit=8):
    '''
    Builds a `PyTorch` model from MIT's custom pkl format pre-trained models
    for 8-bit integer operations

    This is the model that will eventually be used for deployment on MCUs
    '''

    if 'first_conv' in pkl_file: #ProxylessNAS backbone
        first_convolution = construct_q_conv(pkl_file['first_conv'], w_bit=n_bit)
    else:
        raise NotImplementedError
    
    # All blocks in the pkl file
    blocks = tch.nn.Sequential()
    for b in pkl_file['blocks']:
        blocks.append(construct_q_block(b,n_bit=n_bit))

    # feature mix layer if present, else identity layer
    if pkl_file['feature_mix'] is not None:
        feature_mix_convolution = construct_q_conv(pkl_file['feature_mix'], w_bit=n_bit)
    else:
        feature_mix_convolution = tch.nn.Identity()
    
    #average pool
    average_pool = QuantizedAvgPool()

    # FC will always have an 8 bit output
    fc = construct_q_conv(pkl_file['classifier'], w_bit=n_bit, a_bit=8) 
    
    # make net with new units created from the `.pkl` file
    net = tch.nn.Sequential(first_convolution, blocks, feature_mix_convolution, average_pool, fc)
    return net