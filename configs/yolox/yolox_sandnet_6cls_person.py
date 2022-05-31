# fp16 = dict(loss_scale=512.)
# fp16 = dict(loss_scale=dict(init_scale=512))
interval = 10
checkpoint_config = dict(interval=1)
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
data_root = '/usr/videodate/dataset/coco/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)
batch_size = 16
basic_lr_per_img = 3.125e-05
img_scale = (640, 640)

model = dict(
    type='YOLOX',
    backbone=dict(
        type='GeneralSandNet',
        stem_channels=24,
        stage_channels=(32, 36, 48, 64),
        block_per_stage=(1, 2, 4, 4),
        expansion=[1, 4, 4, 4],
        kernel_size=[3, 3, 3, 3],
        num_out=3,
        conv_cfg=dict(type="RepVGGConv")
    ),
    neck=dict(
        type='YOLOXPAFPN_Sep_DBB',
        in_channels=[144, 192, 256],
        out_channels=64,
        num_csp_blocks=1,
        conv_type="RepVGGBlock",
        act_cfg=dict(type='ReLU'),
        dilate=1),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=6,
        in_channels=64,
        feat_channels=64,
        stacked_convs=1,
        act_cfg=dict(type='ReLU')),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))


data = dict(
    samples_per_gpu=20,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=data_root+'annotations/coco_half_person_80_train.json',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        img_prefix=data_root+'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='LoadPasetImages',
                 class_names=["fake_person", "camera"],
                 base_cls_num=3,
                 image_root="./data_paste",
                 to_float32=True,
                 ),
            dict(type='PhotoMetricDistortion', brightness_delta=48),
            # dict(
            #     type='PhotoMetricDistortion',
            #     brightness_delta=32,
            #     contrast_range=(0.5, 1.5),
            #     saturation_range=(0.5, 1.5),
            #     hue_delta=18),
            dict(
                type='Resize',
                img_scale=[(512, 200), (512, 360)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    #     dict(
    #         type='CocoDataset',
    #         ann_file=data_root+'coco_half_person_80_val.json',
    #         classes=['person', 'bottle', 'chair', 'potted plant'],
    #         img_prefix=data_root+'val2017/images',
    #         pipeline=[
    #             dict(type='LoadImageFromFile', to_float32=True),
    #             dict(type='LoadAnnotations', with_bbox=True),
    #             # dict(
    #             #     type='PhotoMetricDistortion',
    #             #     brightness_delta=32,
    #             #     contrast_range=(0.5, 1.5),
    #             #     saturation_range=(0.5, 1.5),
    #             #     hue_delta=18),
    #             dict(
    #                 type='Resize',
    #                 img_scale=[(800, 600), (800, 360)],
    #                 multiscale_mode='range',
    #                 keep_ratio=True),
    #             dict(type='RandomFlip', flip_ratio=0.5),
    #             dict(type='Normalize', **img_norm_cfg),
    #             dict(type='Pad', size_divisor=32),
    #             dict(type='DefaultFormatBundle'),
    #             dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    #     ])
    # ])
    test=dict(
        type='CocoDataset',
        ann_file=data_root+'annotations/coco_half_person_80_val.json',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        img_prefix=data_root+'val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
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
        ann_file=data_root+'annotations/coco_half_person_80_val.json',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        img_prefix=data_root+'val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
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
        ])
)


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

evaluation = dict(interval=1, metric='bbox', classwise=True)
work_dir = './work_dirs/yolox_sandnet_4cl_person_small_neck_24_repvgg_load_from_60e_512_256_2_resolution'
gpu_ids = range(0, 2)