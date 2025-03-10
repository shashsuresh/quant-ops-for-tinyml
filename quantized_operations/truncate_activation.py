import torch as tch

class TruncateActivationRange(tch.autograd.Function):
    '''
    This class defines the custom forward and backward passes
    for the activation range truncation process, to ensure that
    the resultant value is within a pre-determined range, avoiding
    overflow and underflow issues
    '''
    @staticmethod
    def forward(ctx, x, a_bit):
        '''
        Forward pass function for the activation range truncation operation
        '''
        # Save a_bit for later
        ctx.a_bit = a_bit

        # Calculate binary mask
        binary_mask = (-2 ** (a_bit - 1) <= x) & (x <= 2 ** (a_bit - 1) - 1)

        #save mask for later
        ctx.save_for_backward(binary_mask)

        # return the truncated tensor - ensuring all values fit in the chosen lower
        # precision range
        return x.clamp(-2 ** (a_bit - 1), 2 ** (a_bit - 1) - 1)
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        Backward pass function for the activation range truncation operation
        '''

        # Retrieve saved binary mask
        binary_mask = ctx.saved_tensors

        # essentially applies the binary mask to the grad output, to ensure it
        # gives results corresponding to the selected truncation range
        grad_x = grad_output * binary_mask
        
        return grad_x, None