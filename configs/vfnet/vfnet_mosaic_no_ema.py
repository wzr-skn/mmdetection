fp16 = dict(loss_scale=512.)
interval = 10
checkpoint_config = dict(interval=2)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])

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
    type='GFL',
    backbone=dict(
        type='CSPOSANet',
        stem_channels=64,
        stage_channels=(64, 64, 72, 96, 128, 192),
        block_per_stage=(1, 2, 8, 6, 6, 6),
        kernel_size=[3, 3, 3, 3, 3, 3],
        conv_type=dict(
            type='NormalConv',
            info=dict(norm_cfg=dict(type='BN', requires_grad=True))),
        conv1x1=False),
    neck=dict(
        type='YLFPNv2',
        in_channels=[64, 72, 96, 128, 192],
        out_channels=64),
    bbox_head=dict(
        type='VFNetMultiAnchor',
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=4,
        in_channels=64,
        stacked_convs=2,
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
        objectness=True))

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
# test_cfg = dict(
#     nms_pre=1000,
#     min_bbox_size=0,
#     score_thr=0.05,
#     nms=dict(type='nms', iou_threshold=0.3),
#     max_per_img=100)
test_cfg = dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))


data = dict(
    samples_per_gpu=64,
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


runner = dict(type='EpochBasedRunner', max_epochs=120)

evaluation = dict(interval=2, metric='bbox', classwise=True)
work_dir = './work_dirs/vfnet_mosaic_multiscale_30000_test'
gpu_ids = range(0, 2)
