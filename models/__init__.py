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

    if args.version in ['yolof-r18', 'yolof-r50', 'yolof-r50-DC5',
                        'yolof-rt-r50', 'yolof-r101', 'yolof-r101-DC5']:
        return build_yolof(args, cfg, device, num_classes, trainable, pretrained, eval_mode)

    elif args.version in ['fcos-r18', 'fcos-r50', 'fcos-r101',
                          'fcos-rt-r18', 'fcos-rt-r50']:
        return build_fcos(args, cfg, device, num_classes, trainable, pretrained, eval_mode)

    elif args.version in ['retinanet-r18', 'retinanet-r50', 'retinanet-r101',
                          'retinanet-rt-r18', 'retinanet-rt-r50']:
        return build_retinanet(args, cfg, device, num_classes, trainable, pretrained, eval_mode)
