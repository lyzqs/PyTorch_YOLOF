# retinanet config


retinanet_config = {
    'retinanet-r18': {
        # ----------------- PreProcess -----------------
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': False,
        'transforms': [{'name': 'RandomHorizontalFlip'},
                       {'name': 'ToTensor'},
                       {'name': 'Resize'},
                       {'name': 'Normalize'}],
        # ----------------- Network Parameters -----------------
        ## Backbone
        'backbone': 'resnet18',
        'res5_dilation': False,
        'stride': [8, 16, 32, 64, 128],  # P3, P4, P5, P6, P7
        'bk_act_type': 'relu',
        'bk_norm_type': 'FrozeBN',
        ## Neck: FPN
        'fpn': 'basic_fpn',
        'from_c5': True,
        'p6_feat': True,
        'p7_feat': True,
        ## Head
        'head_dim': 256,
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_act_type': 'relu',
        'head_norm_type': 'GN',
        # ----------------- PostProcess -----------------
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # ----------------- Anchor box Configuration -----------------
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # ----------------- Label Assignment -----------------
        'matcher': 'matcher',
        'iou_t': [0.4, 0.5],
        'iou_labels': [0, -1, 1], # [negative sample, ignored sample, positive sample]
        'allow_low_quality_matches': True,
        'ctr_clamp': None,
        # ----------------- Loss Configuration-----------------
        ## Loss hyper-parameters
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training Configuration -----------------
        ## optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'base_lr': 0.01 / 16.,
        'bk_lr_ratio': 1.0,
        ## Warmup
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        ## Epoch
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

    'retinanet-r50': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': False,
        'transforms': [{'name': 'RandomHorizontalFlip'},
                       {'name': 'ToTensor'},
                       {'name': 'Resize'},
                       {'name': 'Normalize'}],
        # model
        'backbone': 'resnet50',
        'res5_dilation': False,
        'stride': [8, 16, 32, 64, 128],  # P3, P4, P5, P6, P7
        'bk_act_type': 'relu',
        'bk_norm_type': 'FrozeBN',
        # fpn neck
        'fpn': 'basic_fpn',
        'from_c5': True,
        'p6_feat': True,
        'p7_feat': True,
        # head
        'head_dim': 256,
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_act_type': 'relu',
        'head_norm_type': 'GN',
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # anchor box
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'matcher': 'matcher',
        'iou_t': [0.4, 0.5],
        'iou_labels': [0, -1, 1], # [negative sample, ignored sample, positive sample]
        'allow_low_quality_matches': True,
        'ctr_clamp': None,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'base_lr': 0.01 / 16.,
        'bk_lr_ratio': 1.0,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

    'retinanet-rt-r18': {
        # input
        'train_min_size': 512,
        'train_max_size': 900,
        'test_min_size': 512,
        'test_max_size': 736,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': False,
        'transforms': [{'name': 'RandomHorizontalFlip'},
                       {'name': 'ToTensor'},
                       {'name': 'Resize'},
                       {'name': 'Normalize'}],
        # model
        'backbone': 'resnet18',
        'res5_dilation': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        'bk_act_type': 'relu',
        'bk_norm_type': 'BN',
        # fpn neck
        'fpn': 'basic_fpn',
        'from_c5': False,
        'p6_feat': False,
        'p7_feat': False,
        # head
        'head_dim': 160,
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_act_type': 'relu',
        'head_norm_type': 'GN',
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # anchor box
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'matcher': 'matcher',
        'iou_t': [0.4, 0.5],
        'iou_labels': [0, -1, 1], # [negative sample, ignored sample, positive sample]
        'allow_low_quality_matches': True,
        'ctr_clamp': None,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'base_lr': 0.01 / 16.,
        'bk_lr_ratio': 1.0,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'epoch': {
            '4x': {'max_epoch': 48, 
                    'lr_epoch': [33, 44], 
                    'multi_scale': [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608]},
        },
    },

    'retinanet-rt-r50': {
        # input
        'train_min_size': 512,
        'train_max_size': 900,
        'test_min_size': 512,
        'test_max_size': 736,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'min_box_size': 8,
        'mosaic': False,
        'transforms': [{'name': 'RandomHorizontalFlip'},
                       {'name': 'ToTensor'},
                       {'name': 'Resize'},
                       {'name': 'Normalize'}],
        # model
        'backbone': 'resnet50',
        'res5_dilation': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        'bk_act_type': 'relu',
        'bk_norm_type': 'BN',
        # fpn neck
        'fpn': 'basic_fpn',
        'from_c5': False,
        'p6_feat': False,
        'p7_feat': False,
        # head
        'head_dim': 160,
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_act_type': 'relu',
        'head_norm_type': 'GN',
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.05,
        'nms_thresh_val': 0.6,
        # anchor box
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'matcher': 'matcher',
        'iou_t': [0.4, 0.5],
        'iou_labels': [0, -1, 1], # [negative sample, ignored sample, positive sample]
        'allow_low_quality_matches': True,
        'ctr_clamp': None,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'base_lr': 0.01 / 16.,
        'bk_lr_ratio': 1.0,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'epoch': {
            '4x': {'max_epoch': 48, 
                    'lr_epoch': [33, 44], 
                    'multi_scale': [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608]},
        },
    },

}