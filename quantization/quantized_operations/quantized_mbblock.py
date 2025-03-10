import torch as tch
from quantized_operations.truncate_activation import TruncateActivationRange

class QuantizedMBBlock(tch.nn.Module):
    '''
    Quantized residual connection block for MobileNet networks
    '''
    def __init__(self, conv, q_add=None, residual_conv=None, a_bit=8):
        super().__init__()
        self.conv = conv
        self.q_add = q_add
        self.residual_conv = residual_conv

        self.a_bit = a_bit

    def forward(self, x):
        '''
        Forward pass for the Quantized Mb Block layer
        '''

       #First apply the convolution
        out = self.conv(x)

        # Handle residual connection if present
        if self.q_add is not None:
            if self.residual_conv(x) is not None:
                x = self.residual_conv(x)

            out = self.q_add(x, out)

            return TruncateActivationRange.apply(out, self.a_bit)
        else:
            return out