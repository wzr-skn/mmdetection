# model settings
dataset_type = 'CocoDataset'
data_root = '/media/traindata_ro/users/yl3076/'

base_lr = 0.01
warmup_iters = 500
# fp16 = dict(loss_scale=512.)
model = dict(
    type='CenterNet',
    backbone=dict(
        type='RepVGGNet',
        stem_channels=32,
        stage_channels=(32, 64, 96, 128),
        block_per_stage=(1, 3, 6, 8),
        kernel_size=[3, 3, 3, 3],
        num_out=1
        # conv_cfg=dict(type="RepVGGConv")
        # conv_type="DBBBlock"
    ),

    neck=dict(
        type='CTResNetNeck',
        in_channel=128,
        num_deconv_filters=(64, 32, 16),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=False),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,
        in_channel=16,
        feat_channel=16,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='SmoothL1Loss', loss_weight=0.05),
        loss_offset=dict(type='SmoothL1Loss', loss_weight=0.5)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))
cudnn_benchmark = True
# dataset settings

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomCenterCropPad',
        crop_size=(320, 320),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(
        type='Resize',
        img_scale=[(320, 320)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
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
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                 keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=
        dict(
        type=dataset_type,
        ann_file=data_root+'ImageDataSets/OpenImageV6_CrowdHuman/OpenImageCrowdHuman_train.json',
        img_prefix=data_root+'ImageDataSets/OpenImageV6_CrowdHuman/WIDER_train/images',
        classes=["person"],
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        ann_file=data_root + '/hollywoodheads/hollywoodhead_val.json',
        img_prefix=data_root + '/hollywoodheads/JPEGImages/',
        classes=["person"],
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root + '/hollywoodheads/hollywoodhead_val.json',
        img_prefix=data_root + '/hollywoodheads/JPEGImages/',
        pipeline=test_pipeline)
            )

evaluation = dict(interval=2, metric='bbox')

optimizer = dict(type='AdamW', lr=0.001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#
# optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)


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
work_dir = './work_dirs/CenterNet_VGG_RepVGG_smoothl1loss_0.5'
load_from = None
resume_from = None
workflow = [('train', 1)]