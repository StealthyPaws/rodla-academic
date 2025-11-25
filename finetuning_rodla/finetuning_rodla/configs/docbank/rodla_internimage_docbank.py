# RoDLA Fine-tuning Configuration for DocBank
# CVPR 2024 - Document Layout Analysis

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# Pre-trained RoDLA weights from PubLayNet
pretrained = 'checkpoints/rodla_internimage_xl_publaynet.pth'

model = dict(
    type='ATSS',
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 22, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.3,  # Reduced for fine-tuning
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=True,
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[384, 768, 1536],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=11,  # DocBank classes
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100)))

# Dataset settings for DocBank
dataset_type = 'CocoDataset'
data_root = 'data/DocBank_coco/'

# DocBank classes
classes = ('abstract', 'author', 'caption', 'equation', 'figure', 
           'footer', 'list', 'paragraph', 'reference', 'section', 'table')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Fine-tuning pipeline (simpler than full training)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',  # Using val for test during fine-tuning
        img_prefix=data_root + 'images/',
        classes=classes,
        pipeline=test_pipeline))

# Fine-tuning optimizer (lower learning rate)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Fine-tuning schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)

# Evaluation and logging
evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# Work directory
work_dir = './work_dirs/rodla_docbank'