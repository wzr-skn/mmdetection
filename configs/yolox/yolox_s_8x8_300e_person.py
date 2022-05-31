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

model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet_no', deepen_factor=0.33, widen_factor=0.4375, conv_cfg=dict(type='RepVGGConv'), conv_type='RepVGGBlock'),
    neck=dict(
        type='YOLOXPAFPN_DBB',
        in_channels=[112, 224, 448],
        out_channels=112,
        num_csp_blocks=1,
        act_cfg=dict(type='ReLU'),
        conv_cfg=dict(type='RepVGGConv'),
        conv_type='RepVGGBlock'),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=4,
        in_channels=112,
        feat_channels=112,
        stacked_convs=1,
        act_cfg=dict(type='ReLU')),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
    input_size=(800, 800),
    random_size_range=(12, 36)
)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=data_root+'coco_half_person_80_train.json',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        img_prefix=data_root+'train2017/images',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PhotoMetricDistortion', brightness_delta=48),
            # dict(
            #     type='PhotoMetricDistortion',
            #     brightness_delta=32,
            #     contrast_range=(0.5, 1.5),
            #     saturation_range=(0.5, 1.5),
            #     hue_delta=18),
            dict(
                type='Resize',
                img_scale=[(1024, 768), (1024, 460)],
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
                img_scale=(1024, 576),
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
                img_scale=(1024, 576),
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
#     dict(type='YOLOXHSVRandomAug'),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Resize', img_scale=img_scale, keep_ratio=True),
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
#         ann_file=data_root+'annotations/instances_train2017.json',
#         img_prefix=data_root+'train2017/images',
#         classes=['person', 'bottle', 'chair', 'potted plant'],
#         pipeline=[
#             dict(type='LoadImageFromFile', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_empty_gt=True,
#     ),
#     pipeline=train_pipeline)
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(800, 448),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=False),
#             dict(type='RandomFlip'),
#             dict(type='Pad', size_divisor=32, pad_val=127.5),
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
#         ann_file=data_root+'annotations/instances_val2017.json',
#         img_prefix=data_root+'val2017/images',
#         classes=['person', 'bottle', 'chair', 'potted plant'],
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/instances_val2017.json',
#         img_prefix=data_root+'val2017/images',
#         pipeline=test_pipeline))

optimizer = dict(type='AdamW', lr=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=0.0005,
    warmup_iters=1,
    min_lr_ratio=0.01)

# lr_config = dict(
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=15,
#     min_lr_ratio=0.05)
#
#
# custom_hooks = [
#     dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
#     dict(
#         type='SyncNormHook',
#         num_last_epochs=15,
#         interval=interval,
#         priority=48),
#     dict(
#         type='ExpMomentumEMAHook',
#         resume_from=resume_from,
#         momentum=0.0001,
#         priority=49)
# ]

runner = dict(type='EpochBasedRunner', max_epochs=120)

evaluation = dict(interval=2, metric='bbox', classwise=True)
# evaluation = dict(
#     save_best='auto',
#     # The evaluation interval is 'interval' when running epoch is
#     # less than ‘max_epochs - num_last_epochs’.
#     # The evaluation interval is 1 when running epoch is greater than
#     # or equal to ‘max_epochs - num_last_epochs’.
#     interval=2,
#     dynamic_intervals=[(105, 1)],
#     metric='bbox')
work_dir = './work_dirs/body_detect/1024x576_medium_yolox_cspdarknet_no_pafpn_RepVGG_load_from_yolox_big_mutiscale'
gpu_ids = range(0, 2)
