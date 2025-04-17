'''
A collection of functions to manipulate a net obtain the desired structure
'''
import torch as tch
import numpy as np
from .quantized_operations.scaled_linear import ScaledLinear

def append_scaled_linear(quantized_model, norm_features=False, num_classes=10):
    '''
    Appends a scaled linear layer and a flatten at the end of a provided model
    '''

    # If last layer is not flatten, append flatten!
    assert isinstance(quantized_model, tch.nn.Sequential)
    if quantized_model[-1] != tch.nn.Flatten:
        quantized_model.append(tch.nn.Flatten())

    # Scaled linear becomes the second last layer
    quantized_model[-2] = ScaledLinear(quantized_model[-2].in_channels, num_classes, quantized_model[-2].x_scale,
                                       quantized_model[-2].zero_x, norm_feat=norm_features)

    return quantized_model

def append_qconv2d_head(quantized_model, num_classes=10):
    from .quantized_operations.quantized_conv2d import QuantizedConv2D

    # If last layer is not flatten, append flatten!
    assert isinstance(quantized_model, tch.nn.Sequential)
    if quantized_model[-1] != tch.nn.Flatten:
        quantized_model.append(tch.nn.Flatten())

    # Create a temporary linear layer, for next calculations
    temp_linear = tch.nn.Conv2d(quantized_model[-2].in_channels, num_classes, 1)

    w_scales_new = get_weight_scales(temp_linear.weight.data, 8)

    weights, bias = get_qw_and_qb(temp_linear.weight.data, temp_linear.bias.data, w_scales_new,
                                  quantized_model[-2].x_scale, 8)
    
    original_operation = quantized_model[-2]

    effective_scale = (quantized_model[-2].x_scale * w_scales_new).float()

    # Create the quantized layer with the quantized weights and bias

    quantized_model[-2] = QuantizedConv2D(quantized_model[-2].in_channels, num_classes, 1,
                                          zero_x=quantized_model[-2].zero_x, zero_y=0,
                                          effective_scale=effective_scale,
                                          w_bit=8, a_bit=8
                                          )
    quantized_model[-2].weight.data = tch.from_numpy(weights).float()
    quantized_model[-2].bias.data = tch.from_numpy(bias).float()
    quantized_model[-2].x_scale = original_operation.x_scale
    quantized_model[-2].y_scale = original_operation.y_scale

    return quantized_model


def get_weight_scales(w, n_bit=8, k_near_zero_tolerance=1e-6, allow_all_same=False):
    '''
    Get the w_scales, for quantizing weights

    IMPORTANT: the zero point for w is always chosen as 0 to make this a symmetric quantization
    '''
    def _extract_min_max_from_weight(weights):
        '''
        Get minimum and maximum weights, so the quantization can be based around this
        '''
        dim_size = weights.shape[0]

        if weights.max() == weights.min():  # all the elements are the same?
            mins = np.zeros(dim_size)
            maxs = np.zeros(dim_size)
            single_value = weights.min().item()
            if single_value < 0.:
                mins[:] = single_value
                maxs[:] = -single_value
            elif single_value > 0.:
                mins[:] = -single_value
                maxs[:] = single_value
            else:
                mins[:] = maxs[:] = single_value
            return tch.from_numpy(mins).to(weights.device), tch.from_numpy(maxs).to(weights.device)
        else:
            weights = weights.reshape(weights.shape[0], -1)
            mins = weights.min(dim=1)[0]
            maxs = weights.max(dim=1)[0]
            maxs = tch.max(mins.abs(), maxs.abs())
            mins = -maxs
            return mins, maxs

    def _expand_very_small_range(mins, maxs):
        '''
        Range expansion, to ensure accuracy is as high as possible
        '''
        k_smallest_half_range = k_near_zero_tolerance / 2
        if (maxs - mins).min() > k_near_zero_tolerance:
            return mins, maxs
        else:
            for i in range(len(mins)):
                mins[i] = min(mins[i], -k_smallest_half_range)
                maxs[i] = max(maxs[i], k_smallest_half_range)
            return mins, maxs

    mins, maxs = _extract_min_max_from_weight(w)
    mins, maxs = _expand_very_small_range(mins, maxs)

    # ensure quantization is symmetric
    assert (mins + maxs).max() < 1e-9  # symmetric

    # return scales
    return maxs / (2 ** (n_bit - 1) - 1)

def get_qw_and_qb(w, b, w_scales, x_scale, n_bit=8):
    '''
    helper function that quantizes weights and biases
    '''
    w = (w / w_scales.view(-1, 1, 1, 1)).round().int()

    assert w.min().item() >= - 2 ** (n_bit - 1) + 1 and w.max().item() <= 2 ** (n_bit - 1) - 1
    w = w.cpu().numpy().astype(np.int8)

    b = (b / w_scales / x_scale).round().int()
    assert b.min().item() >= -2147483648 and b.max().item() <= 2147483647
    b = b.cpu().numpy().astype(np.int32)

    return w, b