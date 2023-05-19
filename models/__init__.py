import torch
from .yolof.build import build_yolof


# build object detector
def build_model(args, 
                cfg, 
                device, 
                num_classes=80, 
                trainable=False, 
                pretrained=None,
                eval_mode=False):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    return build_yolof(args, cfg, device, num_classes, trainable, pretrained, eval_mode)

