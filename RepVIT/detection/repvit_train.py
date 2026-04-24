#!/usr/bin/env python3
"""
RepViT 검출 학습 YOLO-style 런처
반드시 detection/ 디렉토리에서 실행하세요.

사용 예시:
    python repvit_train.py \\
        --config  configs/faster_rcnn_repvit_m1_1_fpn_1x_qnsfl.py \\
        --epochs  12 \\
        --batch   4 \\
        --imgsz   800 \\
        --workers 4 \\
        --device  1 \\
        --patience 5 \\
        --lr      0.0002 \\
        --seed    0 \\
        --project /home/hsjeong/workspace/VIT/RepVIT/detection/results \\
        --name    repvit_m1_1_baseline
"""

import argparse
import os
import os.path as osp
import sys
import time
import warnings

# detection/ 디렉토리를 import 경로에 추가
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import torch
import mmcv
from mmcv import Config
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes, update_data_root)

from mmdet_custom.apis.train import train_detector
import mmcv_custom.runner.epoch_based_runner      # noqa: F401
import mmcv_custom.runner.optimizer               # noqa: F401
import mmcv_custom.runner.early_stopping_hook     # noqa: F401  EarlyStoppingHook 등록
import repvit                                     # noqa: F401  backbone 등록


# ─────────────────────────────────────────────────────────────────────────────
#  CLI 인자 정의
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='RepViT Detection — YOLO-style training launcher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 필수 ──────────────────────────────────────────────────────────────────
    p.add_argument('--config', required=True,
                   help='base config 파일 경로 (detection/ 기준 상대경로)')

    # ── 학습 설정 ─────────────────────────────────────────────────────────────
    p.add_argument('--epochs',    type=int,   default=None,
                   help='총 학습 epoch 수')
    p.add_argument('--batch',     type=int,   default=None,
                   help='전체 배치 크기 (자동으로 GPU 수만큼 나눔)')
    p.add_argument('--imgsz',     type=int,   default=None,
                   help='입력 이미지 크기 (정사각형 기준, 예: 800)')
    p.add_argument('--workers',   type=int,   default=None,
                   help='GPU당 DataLoader 워커 수')
    p.add_argument('--lr',        type=float, default=None,
                   help='기본 학습률 (base learning rate)')
    p.add_argument('--patience',  type=int,   default=None,
                   help='얼리스탑 patience (0 = 비활성)')

    # ── 디바이스 ──────────────────────────────────────────────────────────────
    p.add_argument('--device',    type=str,   default=None,
                   help='사용할 GPU 번호. 단일: "0"  멀티: "0,1"')

    # ── 출력 경로 ─────────────────────────────────────────────────────────────
    p.add_argument('--project',   type=str,   default='work_dirs',
                   help='결과 저장 루트 디렉토리')
    p.add_argument('--name',      type=str,   default='exp',
                   help='실험 이름 (project/name 폴더에 저장)')

    # ── 기타 ──────────────────────────────────────────────────────────────────
    p.add_argument('--seed',      type=int,   default=None,
                   help='재현성을 위한 랜덤 시드')
    p.add_argument('--resume',    type=str,   default=None,
                   help='학습 재개할 체크포인트 경로')
    p.add_argument('--pretrained', type=str,  default=None,
                   help='backbone 사전학습 가중치 경로 오버라이드')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Config 오버라이드
# ─────────────────────────────────────────────────────────────────────────────

def apply_overrides(cfg, args):
    """CLI 인자를 mmdet Config 객체에 반영합니다."""

    # ── GPU / 디바이스 ────────────────────────────────────────────────────────
    if args.device is not None:
        cfg.gpu_ids = [int(g) for g in str(args.device).split(',')]
    num_gpus = len(cfg.gpu_ids)

    # ── 배치 크기 ─────────────────────────────────────────────────────────────
    #   전체 배치 ÷ GPU 수 = GPU당 배치
    if args.batch is not None:
        cfg.data.samples_per_gpu = max(1, args.batch // num_gpus)

    # ── 워커 수 ───────────────────────────────────────────────────────────────
    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers

    # ── Epoch ─────────────────────────────────────────────────────────────────
    if args.epochs is not None:
        cfg.runner.max_epochs = args.epochs
        # LR decay step 을 epoch 비율로 비례 조정 (기본 67%, 92% 지점)
        if hasattr(cfg, 'lr_config') and 'step' in cfg.lr_config:
            cfg.lr_config.step = [
                max(1, int(args.epochs * 0.67)),
                max(1, int(args.epochs * 0.92)),
            ]

    # ── 학습률 ────────────────────────────────────────────────────────────────
    if args.lr is not None:
        cfg.optimizer.lr = args.lr

    # ── 이미지 크기 ───────────────────────────────────────────────────────────
    if args.imgsz is not None:
        sz = args.imgsz
        for step in cfg.data.train.pipeline:
            if step['type'] == 'Resize':
                step['img_scale'] = (sz, sz)
                step.pop('multiscale_mode', None)
                break
        for split in ('val', 'test'):
            for step in getattr(cfg.data, split).pipeline:
                if step.get('type') == 'MultiScaleFlipAug':
                    step['img_scale'] = (sz, sz)
                    break

    # ── 얼리스탑 ──────────────────────────────────────────────────────────────
    if args.patience is not None:
        # 기존 EarlyStoppingHook 제거 후 재등록
        hooks = [h for h in cfg.get('custom_hooks', [])
                 if h.get('type') != 'EarlyStoppingHook']
        if args.patience > 0:
            hooks.append(dict(
                type='EarlyStoppingHook',
                monitor='bbox_mAP',
                patience=args.patience,
                rule='greater',
                min_delta=0.001,
            ))
        cfg.custom_hooks = hooks

    # ── Backbone 사전학습 가중치 ──────────────────────────────────────────────
    if args.pretrained is not None:
        cfg.model.backbone.init_cfg.checkpoint = args.pretrained

    # ── 재개 ──────────────────────────────────────────────────────────────────
    if args.resume is not None:
        cfg.resume_from = args.resume

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  실효 설정 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_settings(cfg, args, logger):
    num_gpus   = len(cfg.gpu_ids)
    total_batch = cfg.data.samples_per_gpu * num_gpus

    patience_val = next(
        (h.get('patience') for h in cfg.get('custom_hooks', [])
         if h.get('type') == 'EarlyStoppingHook'),
        'disabled',
    )

    # train pipeline 에서 실제 적용된 img_scale 추출
    img_scale = '(config 기본값)'
    for step in cfg.data.train.pipeline:
        if step.get('type') == 'Resize':
            img_scale = step.get('img_scale', img_scale)
            break

    logger.info('')
    logger.info('┌─────────────────────────────────────────────────┐')
    logger.info('│         RepViT Detection — 학습 설정 요약          │')
    logger.info('├─────────────────────────────────────────────────┤')
    logger.info(f'│  config    : {args.config}')
    logger.info(f'│  work_dir  : {cfg.work_dir}')
    logger.info(f'│  epochs    : {cfg.runner.max_epochs}')
    logger.info(f'│  batch     : {cfg.data.samples_per_gpu} per GPU'
                f'  ×  {num_gpus} GPU(s)  =  {total_batch} total')
    logger.info(f'│  device    : GPU {cfg.gpu_ids}')
    logger.info(f'│  imgsz     : {img_scale}')
    logger.info(f'│  workers   : {cfg.data.workers_per_gpu} per GPU')
    logger.info(f'│  lr        : {cfg.optimizer.lr}')
    logger.info(f'│  patience  : {patience_val}')
    logger.info(f'│  lr decay  : epoch {cfg.lr_config.get("step", "-")}')
    logger.info(f'│  seed      : {cfg.seed}')
    logger.info(f'│  resume    : {cfg.get("resume_from", "None")}')
    logger.info('└─────────────────────────────────────────────────┘')
    logger.info('')


# ─────────────────────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # work_dir = project/name
    work_dir = osp.join(args.project, args.name)

    # ── Config 로드 및 오버라이드 ──────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    update_data_root(cfg)
    cfg = apply_overrides(cfg, args)
    cfg.work_dir   = work_dir
    cfg.auto_resume = False

    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # ── 디렉토리 / 로거 생성 ──────────────────────────────────────────────────
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file  = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger    = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # ── 시드 ──────────────────────────────────────────────────────────────────
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    set_random_seed(seed)
    cfg.seed = seed

    # ── 설정 출력 ─────────────────────────────────────────────────────────────
    print_settings(cfg, args, logger)

    # ── 모델 / 데이터셋 빌드 ──────────────────────────────────────────────────
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]

    if cfg.checkpoint_config is not None:
        from mmdet import __version__
        from mmcv.utils import get_git_hash
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES,
        )
    model.CLASSES = datasets[0].CLASSES

    # ── 학습 시작 ─────────────────────────────────────────────────────────────
    train_detector(
        model, datasets, cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == '__main__':
    main()
