# =============================================================================
#  Faster R-CNN · RepViT-M1.1-MICA backbone · FPN neck
#  Dataset  : data_qnsfl_new  (person / car / motorcycle / plate_number)
#  변경점   : backbone을 repvit_m1_1_mica 로 교체
#             mica_indices=[8,10,12,22] — Stage3/4 SE 블록 4개에 MICA 적용
# =============================================================================

# ── [A] 핵심 설정 ──────────────────────────────────────────────────────────────
GPU_IDS             = [1]
SAMPLES_PER_GPU     = 2
WORKERS_PER_GPU     = 4

MAX_EPOCHS          = 12
EARLY_STOP_PATIENCE = 5

BASE_LR             = 0.0002
LR_DECAY_EPOCHS     = [8, 11]
WARMUP_ITERS        = 500
WARMUP_RATIO        = 0.001
WEIGHT_DECAY        = 0.05

PRETRAINED_CKPT     = 'pretrain/repvit_m1_1_distill_300e.pth'

ANN_ROOT = '/home/hsjeong/workspace/VIT/RepVIT/detection/data/qnsfl/annotations/'

AUG_FLIP_RATIO      = 0.5
AUG_PHOTO_DISTORT   = True
AUG_MULTISCALE      = True
AUG_SCALE_RANGE     = (640, 1333)

LOG_INTERVAL        = 50
CHECKPOINT_INTERVAL = 1

# ── [B] 모델 구조 ──────────────────────────────────────────────────────────────
_base_ = [
    '_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='repvit_m1_1_mica',
        init_cfg=dict(type='Pretrained', checkpoint=PRETRAINED_CKPT),
        out_indices=[2, 6, 20, 23],
        # MICA 적용 블록: Stage3(256ch) 3개 + Stage4(512ch) 1개
        mica_indices=[8, 10, 12, 22],
        mica_heads=4,
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5,
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=4)
    ),
)

# ── [C] 데이터셋 ───────────────────────────────────────────────────────────────
dataset_type = 'CocoDataset'
classes = ('person', 'car', 'motorcycle', 'plate_number')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

_resize = (
    dict(
        type='Resize',
        img_scale=[(AUG_SCALE_RANGE[1], s) for s in
                   range(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1] + 1, 32)],
        multiscale_mode='value',
        keep_ratio=True,
    ) if AUG_MULTISCALE else
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    _resize,
    dict(type='RandomFlip', flip_ratio=AUG_FLIP_RATIO),
    *(
        [dict(type='PhotoMetricDistortion',
              brightness_delta=32, contrast_range=(0.5, 1.5),
              saturation_range=(0.5, 1.5), hue_delta=18)]
        if AUG_PHOTO_DISTORT else []
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800), flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ]),
]

data = dict(
    samples_per_gpu=SAMPLES_PER_GPU,
    workers_per_gpu=WORKERS_PER_GPU,
    train=dict(type=dataset_type, classes=classes,
               ann_file=ANN_ROOT + 'instances_train.json',
               img_prefix='', pipeline=train_pipeline),
    val=dict(type=dataset_type, classes=classes,
             ann_file=ANN_ROOT + 'instances_val.json',
             img_prefix='', pipeline=test_pipeline),
    test=dict(type=dataset_type, classes=classes,
              ann_file=ANN_ROOT + 'instances_test.json',
              img_prefix='', pipeline=test_pipeline),
)

# ── [D] Optimizer / LR ────────────────────────────────────────────────────────
optimizer = dict(_delete_=True, type='AdamW', lr=BASE_LR, weight_decay=WEIGHT_DECAY)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=WARMUP_ITERS,
    warmup_ratio=WARMUP_RATIO,
    step=LR_DECAY_EPOCHS,
)

runner = dict(type='EpochBasedRunner', max_epochs=MAX_EPOCHS)

# ── [E] Hooks / Logging ───────────────────────────────────────────────────────
evaluation = dict(interval=1, metric='bbox')

_early_stop_hook = dict(
    type='EarlyStoppingHook',
    monitor='bbox_mAP',
    patience=EARLY_STOP_PATIENCE,
    rule='greater',
    min_delta=0.001,
)
custom_hooks = [_early_stop_hook] if EARLY_STOP_PATIENCE is not None else []

checkpoint_config = dict(interval=CHECKPOINT_INTERVAL)
log_config = dict(interval=LOG_INTERVAL, hooks=[dict(type='TextLoggerHook')])
gpu_ids = GPU_IDS
