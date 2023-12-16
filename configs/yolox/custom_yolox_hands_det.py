base_lr = 6e-4
warmup_iters = 1000
model = dict(
    type='CUSTOM_YOLOX',
    backbone=dict(
        type='RepVGGNet',
        stem_channels=32,
        stage_channels=(48, 48, 64, 72),
        block_per_stage=(1, 3, 6, 4),
        kernel_size=[3, 3, 3, 3],
        num_out=4),
    neck=dict(
        type='FuseFPN',
        in_channels=[48, 48, 64, 72],
        out_channels=48,
        conv_cfg=dict(type='RepVGGConv')),
    bbox_head=dict(
        type='Custom_YOLOXHead',
        num_classes=1,
        in_channels=48,
        feat_channels=48,
        stacked_convs=1,
        strides=[4],
        conv_cfg=dict(type='RepVGGConv'),
        act_cfg=dict(type="ReLU")),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),)
    # input_size=(128, 128),
    # random_size_range=(4, 8))


img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(
        type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="ColorJitter"),
    dict(type='CenterBoxCrop',
         max_num=3,
         crop_factor=[3, 1.5, 3, 1.5],
         keep_min_expand=True,
         min_expand=[3, 1.5]),
    dict(type='Pad', pad_to_square=True),
    dict(type='Pad', size_divisor=32, pad_val=127.5),
    dict(
        type='Resize',
        img_scale=[(192, 72), (192, 144)],
        keep_ratio=True),
    dict(type='RandomRadiusBlur', prob=0.2),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# train_pipeline_freihand = [
#     dict(
#         type='LoadImageFromFile', to_float32=True, color_type='color'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type="ColorJitter"),
#     dict(type='CenterBoxCrop', max_num=3, crop_factor=[4, 4, 4, 4]),
#     # dict(type='Pad', pad_to_square=True),
#     dict(type='Pad', size_divisor=32, pad_val=127.5),
#     dict(
#         type='Resize',
#         img_scale=[(160, 64), (160, 128)],
#         keep_ratio=True),
#     dict(type='RandomRadiusBlur', prob=0.2),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Normalize',
#         mean=[127.5, 127.5, 127.5],
#         std=[128, 128, 128],
#         to_rgb=True),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        flip=False,
        img_scale=(512, 256),
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Pad', size_divisor=32, pad_val=127.5),
            dict(type='Pad', pad_to_square=True),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg'),
                keys=['img'])
        ])
]
dataset_type = 'CocoDataset'
data_root = '/media/traindata/'

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=72,
    workers_per_gpu=8,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'hands_datasets/oxford-hand-dataset/training_dataset/training_data/training.json',
            img_prefix=data_root + 'hands_datasets/oxford-hand-dataset/training_dataset/training_data/',
            classes=["hand"],
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file=data_root + 'hands_datasets/oxford-hand-dataset/test_dataset/test_data/test.json',
            img_prefix=data_root + 'hands_datasets/oxford-hand-dataset/test_dataset/test_data/',
            classes=["hand"],
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file=data_root + 'hands_datasets/coco_wholebody/coco_wholebody_train_detect_v1.1.json',
            img_prefix=data_root + 'coco/train2017/images/',
            classes=["hand"],
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file=data_root + 'hands_datasets/coco_wholebody/coco_wholebody_val_detect_v1.1.json',
            img_prefix=data_root + 'coco/val2017/images/',
            classes=["hand"],
            pipeline=train_pipeline),
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'hands_datasets/oxford-hand-dataset/validation_dataset/validation_data/validation.json',
        img_prefix=data_root + 'hands_datasets/oxford-hand-dataset/validation_dataset/validation_data/',
        classes=["hand"],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'hands_datasets/oxford-hand-dataset/validation_dataset/validation_data/validation.json',
        img_prefix=data_root + 'hands_datasets/oxford-hand-dataset/validation_dataset/validation_data/',
        pipeline=test_pipeline))
# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
evaluation = dict(interval=2, metric='bbox', classwise=True)

optimizer = dict(type='AdamW', lr=base_lr)
optimizer_config = dict(grad_clip=None)

# optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='Cyclic',
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001,
    cyclic_times=2)
# total_epochs = 120
runner = dict(type='EpochBasedRunner', max_epochs=120)

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
work_dir = './work_dirs/hand_detect/yolox_repvgg_1output_noobj_2value_oxford_cocowholebodyv1.1_small_centerbox'
load_from = './work_dirs/hand_detect/yolox_repvgg_1output_noobj_2value_oxford_cocowholebodyv1.1_centerbox/latest.pth'
resume_from = None
workflow = [('train', 1)]
