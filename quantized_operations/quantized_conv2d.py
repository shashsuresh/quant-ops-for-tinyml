import torch as tch
from quantized_operations.truncate_activation import TruncateActivationRange
from conversions import CONV_W_GRAD, to_pytorch_tensor
    
class QuantizedConv2DFunc(tch.autograd.Function):
    '''
    This class defines the custom forward and backward passes
    of a `QuantizedConv2DFunc` layer
    '''
    @staticmethod
    def forward(ctx, x, weight, bias, zero_x, zero_y, effective_scale, stride, padding, dilation, groups):
        '''
        Computes the forward pass of a quantized 2D convolution layer
        '''

        #ensure x and weights are int only
        x = x.round()
        w = weight.round()

        #save values for later
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = x.shape
        ctx.weight_size = weight.shape
        
        x = x - zero_x

        # Save other tensors based on what mode we are running
        if CONV_W_GRAD:
            ctx.save_for_backward(weight, effective_scale, x)
        else:
            ctx.save_for_backward(weight, effective_scale)

        # Apply 2d convolution operation
        out = tch.nn.functional.F.conv2d(x, weight, None, stride, padding, dilation, groups)

        # output must be int
        out = out.round()

        # bias is saved in memory as f32 and used as an int (after rounding) during inference

        # Reshape bias tensor, so it added correctly
        out = out + bias.view(1, -1, 1, 1)

        # Apply scale and then round
        out = (out * effective_scale.view(1, -1, 1, 1)).round()

        return out + zero_y
    
    def backward(ctx, grad_output):
        '''
        Computes the backward pass of a quantized 2D convolution layer
        '''

        if CONV_W_GRAD:
            weight, effective_scale, x = ctx.saved_tensors
        else:
            weight, effective_scale = ctx.saved_tensors
        
        # calculate gradient of zero_y
        grad_zero_y = grad_output.sum([0, 2, 3])
        
        # calculate the convolution output's gradient wrt loss
        grad_conv_out = grad_output * effective_scale.view([1, -1, 1, 1])

        # calculate the bias's gradient using the convolution output gradient
        grad_bias = grad_conv_out.sum([0, 2, 3])

        # calculate the gradient wrt input tensor
        grad_conv_in = tch.nn.grad.conv2d_input(ctx.input_size, weight, grad_conv_out, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)

        # calculate gradient of zero_x
        grad_zero_x = grad_conv_in.sum([0, 2, 3])

        if CONV_W_GRAD:
            # calculate gradient of weights
            grad_w = tch.nn.grad.conv2d_weight(x, ctx.weight_size, grad_conv_out, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
        else:
            grad_w = None
        
        # TODO check if we need per-channel quantization... - probably necessary when running simulation

        return grad_conv_in, grad_w, grad_bias, grad_zero_x, grad_zero_y, None, None, None, None, None
    
class QuantizedConv2D(tch.nn.Conv2d):
    '''
    A quantized Conv2D layer with custom forward and backward passes
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, 
                dilation = 1, groups = 1, bias = True, padding_mode = "zeros",
                zero_x=0, zero_w=0, zero_y=0, effective_scale=None,
                significand=1, channel_shift=0, w_bit=0, a_bit=None
                ):
        '''
        Initialize a QuantizedConv2D layer
        '''
        super(QuantizedConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.significand = significand

        self.channel_shift = channel_shift

        ## Register these new parameters, as they will be used for later operations
        self.register_buffer('zero_x', to_pytorch_tensor(zero_x))
        self.register_buffer('zero_y', to_pytorch_tensor(zero_y))
        # TODO check if we need scale training
        self.register_buffer('effective_scale', effective_scale)
        
        self.w_bit = w_bit
        self.a_bit = a_bit if a_bit is not None else w_bit
    
    def forward(self, x):
        '''
        The forward function for `QuantizedConv2D`, which calls the forward/backward function
        defined in the `QuantizedConv2DFunc` class.
        '''
        # Apply quantized convolution fn
        out = QuantizedConv2DFunc.apply(x, self.weight, self.bias, self.zero_x, self.zero_y, self.effective_scale, self.stride, self.padding, self.dilation, self.groups)
        #Truncate result
        return TruncateActivationRange.apply(out, self.a_bit)
    