base_lr = 3e-4
warmup_iters = 1000
model = dict(
    type='YOLOX',
    # backbone=dict(
    #     type='GeneralSandNet',
    #     stem_channels=32,
    #     stage_channels=(32, 32, 36, 36),
    #     block_per_stage=(1, 1, 3, 3),
    #     expansion=[1, 2, 3, 3],
    #     kernel_size=[3, 3, 3, 3],
    #     num_out=4,
    # ),
    # neck=dict(
    #     type='FuseFPN',
    #     in_channels=[32, 64, 108, 108],
    #     conv_cfg=dict(type="DepthWiseConv",),
    #     out_channels=32,
    # ),
    backbone=dict(
        type='GeneralSandNet',
        stem_channels=16,
        stage_channels=(24, 32, 32, 36),
        block_per_stage=(1, 1, 3, 3),
        expansion=[1, 2, 3, 3],
        kernel_size=[3, 3, 3, 3],
        num_out=4,
    ),
    neck=dict(
        type='FuseFPN',
        in_channels=[24, 64, 96, 108],
        conv_cfg=dict(type="DepthWiseConv", ),
        out_channels=32),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        strides=[4],
        in_channels=32,
        feat_channels=16,
        use_depthwise=True,
        stacked_convs=1,
        act_cfg=dict(type="ReLU")),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(
        type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="ColorJitter"),
    dict(type='CenterBoxCrop', max_num=3, crop_factor=[4, 4, 4, 4]),
    dict(type='Pad', pad_to_square=True),
    dict(
        type='Resize',
        img_scale=[(320, 128), (320, 256)],
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
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        flip=False,
        img_scale=(128, 128),
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
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
data_root = '/usr/videodate/yehc/'

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=8,
    train=[dict(
        type=dataset_type,
        ann_file=data_root +
        'ImageDataSets/OpenImageV6_CrowdHuman/annotation_crowd_head_train.json',
        img_prefix=data_root +
        'ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images',
        classes=["person"],
        pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file=data_root +
                     'ImageDataSets/SCUT_HEAD/SCUT_HEAD_Part_B/trainval.json',
            img_prefix=data_root +
                     'ImageDataSets/SCUT_HEAD/SCUT_HEAD_Part_B/',
            classes=["person"],
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file=data_root +
                     'ImageDataSets/SCUT_HEAD/SCUT_HEAD_Part_A/trainval.json',
            img_prefix=data_root +
                       'ImageDataSets/SCUT_HEAD/SCUT_HEAD_Part_A/',
            classes=["person"],
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file=data_root +
                     'ImageDataSets/brainwash/trainbrainwash_.json',
            img_prefix=data_root +
                      'ImageDataSets/brainwash/',
            classes=["person"],
            pipeline=train_pipeline),
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        '/hollywoodheads/hollywoodhead_val.json',
        img_prefix=data_root +
        '/hollywoodheads/JPEGImages/',
        classes=["person"],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        '/hollywoodheads/hollywoodhead_val.json',
        img_prefix=data_root +
        '/hollywoodheads/JPEGImages/',
        pipeline=test_pipeline))
# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
evaluation = dict(start=9999, interval=2, metric='bbox', classwise=True)

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
work_dir = './tools/work_dirs/yolox_genereal_dataset_bbox_crop_128_384_trainblur_3datasets'
load_from = './tools/work_dirs/yolox_genereal_dataset_bbox_crop_128_384_trainblur_3datasets/latest.pth'
resume_from = None
workflow = [('train', 1)]
