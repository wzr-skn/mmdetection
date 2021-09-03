base_lr = 1e-2
warmup_iters = 1000

data_root = '/usr/videodate/dataset/coco/'

model = dict(
    type='CenterNet',
    backbone=dict(
        type='SandNet',
        stem_channels=16,
        stage_channels=(16, 24, 28, 32),
        block_per_stage=(1, 3, 6, 3),
        expansion=[1, 2, 4, 4],
        kernel_size=[3, 3, 3, 3],
        num_out=4,
    ),
    neck=dict(
        type='CTResNetNeck',
        in_channel=128,
        num_deconv_filters=(64, 32, 16),
        num_deconv_kernels=(4, 4, 4),
        groups=[128, 64, 32],
        use_dcn=False),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=4,
        in_channel=16,
        groups=16,
        feat_channel=16,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]


# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + "annotations/coco_half_person_80_train.json",
        img_prefix=data_root + 'train2017',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        filter_empty_gt=True,
        pipeline=train_pipeline),

    val=dict(
        type='CocoDataset',
        ann_file='/usr/videodate/dataset/coco/annotations/coco_half_person_80_val.json',
        img_prefix='/usr/videodate/dataset/coco/val2017/',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        filter_empty_gt=True,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file='/usr/videodate/dataset/coco/annotations/coco_half_person_80_val.json',
        img_prefix='/usr/videodate/dataset/coco/val2017/',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        filter_empty_gt=True,
        pipeline=test_pipeline))
# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
evaluation = dict(interval=3, metric='bbox', classwise=True)

optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=None)

# optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)
total_epochs = 120

checkpoint_config = dict(interval=2)


log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
# runtime settings


device_ids = range(1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'tools/work_dirs/centernet_dbb_half_person'
load_from = None
resume_from = 'tools/work_dirs/centernet_dbb_half_person/latest.pth'
workflow = [('train', 1)]