# =============================================================================
#  Faster R-CNN  ·  RepViT-M1.1 backbone  ·  FPN neck
#  Dataset  : data_qnsfl_new  (person / car / motorcycle / plate_number)
#  Framework: mmdetection 2.x
# =============================================================================


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [A]  자주 바꾸는 핵심 설정 — 여기만 수정하면 됩니다
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── [A-1] GPU / 배치 설정 ─────────────────────────────────────────────────────
GPU_IDS            = [1]          # 사용할 GPU 번호 목록.  단일 GPU: [0],  멀티: [0,1]
SAMPLES_PER_GPU    = 2            # GPU 1장당 배치 크기.  OOM 발생 시 1로 줄이세요
WORKERS_PER_GPU    = 4            # 데이터 로딩 워커 수.  CPU 코어 수에 맞게 조정
#
#  실효 배치 크기 = SAMPLES_PER_GPU × len(GPU_IDS)
#  예) GPU 1장 + SAMPLES_PER_GPU=2  →  실효 배치 2
#      GPU 2장 + SAMPLES_PER_GPU=2  →  실효 배치 4  (LR 도 비례해서 높이는 것 권장)

# ── [A-2] 학습 epoch / 얼리스탑 ──────────────────────────────────────────────
MAX_EPOCHS         = 12           # 총 학습 epoch 수
EARLY_STOP_PATIENCE = 5           # val mAP 가 이 epoch 수 동안 개선 없으면 조기 종료
                                  # 얼리스탑 비활성화: None 으로 설정

# ── [A-3] 학습률(LR) 설정 ─────────────────────────────────────────────────────
BASE_LR            = 0.0002       # 기본 학습률
LR_DECAY_EPOCHS    = [8, 11]      # 이 epoch 마다 LR × 0.1 감소
WARMUP_ITERS       = 500          # 처음 N 스텝 동안 LR 을 0 → BASE_LR 로 워밍업
WARMUP_RATIO       = 0.001        # 워밍업 시작 LR = BASE_LR × WARMUP_RATIO
WEIGHT_DECAY       = 0.05         # AdamW weight decay

# ── [A-4] 사전학습 가중치 경로 ────────────────────────────────────────────────
PRETRAINED_CKPT    = 'pretrain/repvit_m1_1_distill_300e.pth'
#                    detection/ 기준 상대경로.  다운로드:
#                    https://github.com/THU-MIG/RepViT/releases

# ── [A-5] 데이터셋 경로 ──────────────────────────────────────────────────────
ANN_ROOT = '/home/hsjeong/workspace/VIT/RepVIT/detection/data/qnsfl/annotations/'
#           COCO JSON 파일들이 저장된 디렉토리.  convert_yolo_to_coco.py 가 생성함
#           img_prefix = '' : JSON 내 file_name 이 절대경로로 저장되어 있음

# ── [A-6] 학습 이미지 Augmentation ───────────────────────────────────────────
#   아래 값만 바꾸면 train_pipeline 에 자동 반영됩니다
AUG_FLIP_RATIO      = 0.5         # 좌우 반전 확률 (0.0 = 비활성)

AUG_PHOTO_DISTORT   = True        # 밝기/대비/채도/색조 랜덤 변환 (True/False)
#   세부 범위 조정은 섹션 C 의 PhotoMetricDistortion 블록에서 수정

AUG_MULTISCALE      = True        # 멀티스케일 리사이즈 (True/False)
#   True  : (640~1333) 범위에서 랜덤하게 단축 길이 선택 → 스케일 다양성 확보
#   False : 고정 스케일 img_scale=(1333, 800) 만 사용
AUG_SCALE_RANGE     = (640, 1333) # AUG_MULTISCALE=True 일 때 단축 길이 범위

# ── [A-7] 로그 / 체크포인트 저장 간격 ────────────────────────────────────────
LOG_INTERVAL        = 50          # N 스텝마다 loss 출력
CHECKPOINT_INTERVAL = 1           # N epoch 마다 체크포인트 저장


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [B]  모델 구조 — backbone / neck / head
#       (RepViT-M1.1 을 바꾸려면 이 블록만 수정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_base_ = [
    '_base_/models/faster_rcnn_r50_fpn.py',  # bbox-only (mask 없음)
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py',
]

model = dict(
    # ── Backbone : RepViT-M1.1 ─────────────────────────────────────────────
    #   _delete_=True : 부모 config 의 ResNet-50 을 완전히 교체
    #   out_indices   : FPN 에 넘길 stage 출력 인덱스 (stage1/2/3/4)
    #   채널           : 64 → 128 → 256 → 512
    backbone=dict(
        _delete_=True,
        type='repvit_m1_1',
        init_cfg=dict(type='Pretrained', checkpoint=PRETRAINED_CKPT),
        out_indices=[2, 6, 20, 24],
    ),

    # ── Neck : FPN ─────────────────────────────────────────────────────────
    #   in_channels 는 backbone out_indices 의 채널과 반드시 일치해야 함
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5,
    ),

    # ── Head : Faster R-CNN RoI Head ───────────────────────────────────────
    #   num_classes : 클래스 수 (background 제외)
    roi_head=dict(
        bbox_head=dict(
            num_classes=4,   # person / car / motorcycle / plate_number
        )
    ),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [C]  데이터셋 설정
#       (클래스명, 파이프라인 변경이 필요하면 이 블록 수정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

dataset_type = 'CocoDataset'
classes      = ('person', 'car', 'motorcycle', 'plate_number')

# ImageNet 정규화 (RGB 순서)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ── Resize 전략 : AUG_MULTISCALE 에 따라 분기 ────────────────────────────────
_resize = (
    dict(                                          # 멀티스케일: 단축 랜덤 선택
        type='Resize',
        img_scale=[(AUG_SCALE_RANGE[1], s) for s in
                   range(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1] + 1, 32)],
        multiscale_mode='value',
        keep_ratio=True,
    )
    if AUG_MULTISCALE else
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)  # 고정 스케일
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    # [1] 리사이즈 (고정 or 멀티스케일)
    _resize,

    # [2] 좌우 반전 — AUG_FLIP_RATIO=0.0 이면 사실상 비활성
    dict(type='RandomFlip', flip_ratio=AUG_FLIP_RATIO),

    # [3] 색상 왜곡 — AUG_PHOTO_DISTORT=False 이면 건너뜀
    #     brightness_delta : 밝기 ±32
    #     contrast_range   : 대비 0.5~1.5
    #     saturation_range : 채도 0.5~1.5
    #     hue_delta        : 색조 ±18
    *(
        [dict(
            type='PhotoMetricDistortion',
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
        )]
        if AUG_PHOTO_DISTORT else []
    ),

    # [4] 정규화 / 패딩 / 포맷 변환 (고정)
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
        ],
    ),
]

data = dict(
    samples_per_gpu=SAMPLES_PER_GPU,
    workers_per_gpu=WORKERS_PER_GPU,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ANN_ROOT + 'instances_train.json',
        img_prefix='',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ANN_ROOT + 'instances_val.json',
        img_prefix='',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ANN_ROOT + 'instances_test.json',
        img_prefix='',
        pipeline=test_pipeline,
    ),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [D]  Optimizer / LR Schedule
#       (섹션 A 변수를 수정하면 자동 반영됨)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=WARMUP_ITERS,
    warmup_ratio=WARMUP_RATIO,
    step=LR_DECAY_EPOCHS,
)

runner = dict(type='EpochBasedRunner', max_epochs=MAX_EPOCHS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  [E]  Evaluation / EarlyStopping / Logging / Checkpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# val mAP 기준으로 매 epoch 평가
evaluation = dict(interval=1, metric='bbox')

# 얼리스탑 훅 (EARLY_STOP_PATIENCE = None 이면 비활성)
_early_stop_hook = dict(
    type='EarlyStoppingHook',
    monitor='bbox_mAP',         # 모니터링 지표: val bbox mAP
    patience=EARLY_STOP_PATIENCE,
    rule='greater',             # 클수록 좋음
    min_delta=0.001,            # 이 값 이상 개선돼야 "개선"으로 인정
)

custom_hooks = (
    [_early_stop_hook] if EARLY_STOP_PATIENCE is not None else []
)

# 체크포인트 저장
checkpoint_config = dict(interval=CHECKPOINT_INTERVAL)

# 로그 출력
log_config = dict(
    interval=LOG_INTERVAL,
    hooks=[dict(type='TextLoggerHook')],
)

# GPU 디바이스 지정 (train.py CLI --gpu-id 와 동일한 역할, 여기서 기본값 설정)
gpu_ids = GPU_IDS
