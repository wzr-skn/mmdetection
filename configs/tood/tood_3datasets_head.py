base_lr = 3e-4
warmup_iters = 1000
model = dict(
    type='TOOD',
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
        out_channels=64,
        conv_cfg=dict(type='RepVGGConv')),
    bbox_head=dict(
        type='TOODHead',
        num_classes=1,
        in_channels=64,
        stacked_convs=6,
        feat_channels=64,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(
        type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="ColorJitter"),
    dict(type='CenterBoxCrop', max_num=3, crop_factor=[4, 4, 4, 4]),
    # dict(type='Pad', pad_to_square=True),
    dict(type='Pad', size_divisor=32, pad_val=127.5),
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
        img_scale=(320, 320),
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
data_root = '/media/traindata_ro/users/yl3076/'

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=[dict(
        type=dataset_type,
        # ann_file=data_root +
        # 'ImageDataSets/OpenImageV6_CrowdHuman/OpenImageCrowdHuman_train.json',
        ann_file='/home/ubuntu/my_datasets/OpenImageV6_CrowdHuman/annotation_crowd_head_train.json',
        img_prefix=data_root +
        'ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images',
        classes=["person"],
        pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_B/trainval.json',
            img_prefix='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_B/',
            classes=["person"],
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_A/trainval.json',
            img_prefix='/home/ubuntu/my_datasets/SCUT_HEAD/SCUT_HEAD_Part_A/',
            classes=["person"],
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            ann_file='/home/ubuntu/my_datasets/brainwash/trainbrainwash_.json',
            img_prefix='/home/ubuntu/my_datasets/brainwash/',
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
work_dir = './work_dirs/optimize_head_detect/tood_1output_4_RepVGG_3datasets_filter_big_scale_no_keep'
load_from = None
resume_from = None
workflow = [('train', 1)]
