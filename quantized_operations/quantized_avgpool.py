import torch as tch
from conversions import round_tensor

class QuantizedAvgPoolFn(tch.autograd.Function):
    '''
    This class defines the custom forward and backward passes
    of a `QuantizedAvgPool` layer
    '''
    @staticmethod
    def forward(ctx, x):
        '''
        Computes the forward pass of a quantized element-wise average pooling layer
        '''

        #Save input dimensions for later
        ctx.input_shape = x.size()

        #Ensure that the datatypes match
        assert x.dtype == tch.float32

        #Avg pool using dimension mean
        #Round result to ensure we can cast it to int later
        return round_tensor(x.mean([-1, -2], keepdim=True))
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        Computes the backward pass of a quantized element-wise average pooling layer
        '''

        # Mathematical representation:
        #   delta_in = (delta_out)/k^2
        #   Where delta out is the gradient of the loss with wrt the output of the previous layer
        #   and k is the size of the pooling window

        # Get input shape from the context
        input_shape = ctx.input_shape
        
        # Get input gradient from the output gradient
        return grad_output.repeat(1, 1, *input_shape[-2:]) / input_shape[-1] * input_shape[-2]
    
class QuantizedAvgPool(tch.nn.Module):
    '''
    A quantized average pool layer with custom forward and backward passes
    '''
    def __init__(self):
        super(QuantizedAvgPool,self).__init__()

    def forward(self, x):
        '''
        The forward function for `QuantizedAvgPool`, which calls the forward/backward function
        defined in the `QuantizedAvgPoolFn` class.
        '''
        return QuantizedAvgPoolFn.apply(x)