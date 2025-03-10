import torch as tch
from conversions import to_pytorch_tensor

class QuantizedElementwiseAddFn(tch.autograd.Function):
    '''
    This class defines the custom forward and backward passes
    of a `QuantizedElementwiseAdd` layer
    '''
    @staticmethod
    def forward(ctx, x1, x2, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        '''
        Computes the forward pass of a quantized element-wise addition layer
        '''
        # x1 and x2 must be ints, round, ensures this is the case
        x1 = x1.round()
        x2 = x2.round()

        # Ensure dimensions match for element wise operation!
        assert x1.size() == x2.size()

        # Save parameters for backward pass
        ctx.save_for_backward(scale_x1, scale_x2, scale_y)

        # Quantize both x1 and x2
        x1 = (x1 - zero_x1) * scale_x1
        x2 = (x2 - zero_x2) * scale_x2

        # perform element wise addition
        out = x1 + x2

        # rescale output tensor
        return (out / scale_y).round() + zero_y
    
    def backward(ctx, grad_output):
        '''
        Computes the backward pass of a quantized element-wise addition layer
        '''
        
        #Retrieve saved scales
        scale_x1, scale_x2, scale_y = ctx.saved_tensors

        # gradient of zero point of y
        grad_zero_y = grad_output.sum([0, 2, 3])

        # gradient of sum wrt loss
        grad_sum = grad_output / scale_y.item()

        # Compute gradients of x1 and x2 wrt loss and scale
        grad_x1 = grad_sum * scale_x1.item()
        grad_x2 = grad_sum * scale_x2.item()

        # Compute gradients of zero points of x1 and x2
        grad_zero_x1 = -grad_x1.sum([0, 2, 3])
        grad_zero_x2 = -grad_x2.sum([0, 2, 3])

        # Only the gradients of zero_y, zero_x1, zero_x2, x1 and x2 are returned
        return grad_x1, grad_x2, grad_zero_x1, grad_zero_x2, grad_zero_y, None, None, None

class QuantizedElementWise(tch.nn.Module):
    '''
    A element-wise addition operation layer designed for quantized operands
    '''
    def __init__(self, operator, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        '''
            Initialize the module and register a few buffers that will later be used 
            for calculations.
        '''
        # Init the parent class
        super().__init__()
        # Set the class variable "operator"
        self.operator = operator
        # **We only support addition at present**
        assert operator == 'add'

        # Register buffers on creation
        self.register_buffer('zero_x1', to_pytorch_tensor(zero_x1))
        self.register_buffer('zero_x2', to_pytorch_tensor(zero_x2))
        self.register_buffer('zero_y', to_pytorch_tensor(zero_y))
        self.register_buffer('scale_x1', to_pytorch_tensor(scale_x1))
        self.register_buffer('scale_x2', to_pytorch_tensor(scale_x2))
        self.register_buffer('scale_y', to_pytorch_tensor(scale_y))

    def forward(self, x1, x2):
        '''
        The forward function for `QuantizedElementWise`, which calls the forward/backward function
        defined in the `QuantizedElementwiseAddFn` class.
        '''
        return QuantizedElementwiseAddFn.apply(x1, x2, self.zero_x1,  self.zero_x2,  self.zero_y,  self.scale_x1, self.scale_x2, self.scale_y)
    