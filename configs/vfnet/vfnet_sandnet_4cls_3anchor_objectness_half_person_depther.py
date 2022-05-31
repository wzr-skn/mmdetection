dataset_type = 'CocoDataset'
data_root = '/media/traindata/coco/'
base_lr = 0.01
warmup_iters = 2000
model = dict(
    type='VFNet',
    backbone=dict(
        type='GeneralSandNet',
        stem_channels=32,
        stage_channels=(32, 36, 48, 64, 64, 72),
        block_per_stage=(1, 2, 4, 4, 1, 1),
        expansion=[1, 4, 4, 4, 4, 4],
        kernel_size=[3, 3, 3, 3, 3, 3],
        conv_cfg=dict(type='RepVGGConv'),
        num_out=5),
    neck=dict(
        type='YLFPNv2',
        in_channels=[144, 192, 256, 256, 288],
        out_channels=64,
        conv_cfg=dict(type='SepConv')),
    bbox_head=dict(
        type='VFNetMultiAnchor',
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=4,
        in_channels=64,
        stacked_convs=1,
        feat_channels=64,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        denorm=True,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=1.5),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            octave_base_scale=6,
            scales_per_octave=1,
            center_offset=0.0,
            strides=[8, 16, 32, 64, 128]),
        loss_bbox_refine=dict(type='CIoULoss', loss_weight=2.0),
        objectness=True),
    train_cfg = dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))
train_pipline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(512, 240), (512, 320)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='PhotoMetricDistortion', brightness_delta=48),
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
val_pipline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=108,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root+'coco_half_person_80_train.json',
        img_prefix=data_root+'train2017/images',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(512, 240), (512, 320)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='PhotoMetricDistortion', brightness_delta=48),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        filter_empty_gt=True),
    val=dict(
        type='CocoDataset',
        ann_file=data_root+'coco_half_person_80_val.json',
        img_prefix=data_root+'val2017/images',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128, 128, 128],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        filter_empty_gt=True),
    test=dict(
        type='CocoDataset',
        ann_file=data_root+'coco_half_person_80_val.json',
        img_prefix=data_root+'val2017/images',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128, 128, 128],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        filter_empty_gt=True))
evaluation = dict(interval=2, metric='bbox', classwise=True)
optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=0.0005,
    warmup_iters=1,
    min_lr_ratio=0.01)
total_epochs = 60
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/vfnet_sandnet_4cls_3anchor_objectness_half_person_depther_load_from_512_256_1'
load_from = './work_dirs/vfnet_sandnet_4cls_3anchor_objectness_half_person_depther_load_from_cosine/latest_previous.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
