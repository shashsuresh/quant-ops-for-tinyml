## Some global configs
import os
import warnings

if os.path.exists("../quant_configs.yml"):
    with open ("../quant_configs.yml") as conf:
        import yaml
        configs = yaml.load(conf, Loader=yaml.Loader)
        QUANTIZED_GRADIENT = configs['q_grad']
        ROUNDING = configs['rounding']
        CONV_W_GRAD = configs['conv_w_grad']
        QUANTIZE_GRADS = configs['quantize_grads']
        TRAIN_SCALES = configs['train_scales']
        print(QUANTIZE_GRADS)
else:
    warnings.warn("No config specified, using default values!")
    ## Default values
    QUANTIZED_GRADIENT = False
    ROUNDING = 'round'
    CONV_W_GRAD = True
    QUANTIZE_GRADS = False
    TRAIN_SCALES = False