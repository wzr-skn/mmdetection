# fp16 = dict(loss_scale=512.)
# fp16 = dict(loss_scale=dict(init_scale=512))
interval = 10
checkpoint_config = dict(interval=2)
log_config = dict(interval=20, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'CocoDataset'
data_root = '/media/traindata/coco/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)
batch_size = 16
basic_lr_per_img = 3.125e-05
img_scale = (640, 640)

# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=50,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=4,
        in_channels=50,
        stacked_convs=4,
        feat_channels=50,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


data = dict(
    samples_per_gpu=10,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file='/home/ubuntu/json/coco_half_person_80_train_load.json',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        img_prefix=data_root+'train2017/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(240, 480), (480, 960)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=data_root+'coco_half_person_80_val.json',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        img_prefix=data_root+'val2017/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32, pad_val=127.5),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128, 128, 128],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file=data_root+'coco_half_person_80_val.json',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        img_prefix=data_root+'val2017/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32, pad_val=127.5),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128, 128, 128],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

# train_pipeline = [
#     dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     dict(
#         type='MixUp',
#         img_scale=img_scale,
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Resize', keep_ratio=True),
#     dict(type='Pad', pad_to_square=True, pad_val=114.0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
#
# train_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         ann_file='/home/ubuntu/json/coco_half_person_80_train_load.json',
#         img_prefix=data_root+'train2017/images',
#         classes=['person', 'bottle', 'chair', 'potted plant'],
#         pipeline=[
#             dict(type='LoadImageFromFile', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_empty_gt=True,
#     ),
#     pipeline=train_pipeline,
#     dynamic_scale=img_scale)
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Pad', size=img_scale, pad_val=114.0),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
#
# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=8,
#     train=train_dataset,
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_half_person_80_val.json',
#         img_prefix=data_root+'val2017/images',
#         classes=['person', 'bottle', 'chair', 'potted plant'],
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_half_person_80_val.json',
#         img_prefix=data_root+'val2017/images',
#         pipeline=test_pipeline))

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

# custom_hooks = [
#     dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
#     dict(
#         type='SyncRandomSizeHook',
#         ratio_range=(14, 26),
#         img_scale=img_scale,
#         interval=interval,
#         priority=48),
#     dict(
#         type='SyncNormHook',
#         num_last_epochs=15,
#         interval=interval,
#         priority=48),
#     dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
# ]

runner = dict(type='EpochBasedRunner', max_epochs=120)

evaluation = dict(interval=2, metric='bbox', classwise=True)
work_dir = './work_dirs/30000_yolox_cspdarknet_no_widen_factor0.375'
gpu_ids = range(0, 2)
