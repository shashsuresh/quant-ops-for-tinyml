import torch as tch
from quantization.conversions import to_pytorch_tensor

class ScaledLinear(tch.nn.Linear):
    '''
    A floating point linear layer (fully-connected layer) used exclusively for training
    '''

    def __init__(self, in_features: int, out_features: int, scale_x, zero_x, bias: bool=True, device=None, dtype=None, norm_feat=False):
        super.__init__(in_features, out_features, bias, device, dtype)
        
        # Register few values for later use
        self.register_buffer('scale_x', to_pytorch_tensor(scale_x))
        self.register_buffer('zero_x', to_pytorch_tensor(zero_x))

        self.norm_feat = norm_feat

        if norm_feat:
            # Fill bias
            self.bias.data.fill_(2.)
            # set epsilon
            self.eps = 1e-5

    def forward(self, x):
        '''
        Forward pass function for the ScaledLinear layer
        '''

        # Dequantize x
        x = (x.squeeze(-1).squeeze(-1) - self.zero_x.detach().view(1, -1)) * self.scale_x.detach().view(1, -1)

        if self.norm_feat:
            #normalize x and weights
            x_norm = x.div(tch.norm(x, p=2, dim=1).view(-1, 1) + self.eps)
            weights_norm = self.weight.div(tch.norm(self.weight, p=2, dim=1).view(-1, 1) + self.eps)

            cosine_dist = (x_norm @ weights_norm.T) * self.bias.view(1, -1)
            return cosine_dist
        else:
            # just apply a regular linear layer
            return super().forward(x)