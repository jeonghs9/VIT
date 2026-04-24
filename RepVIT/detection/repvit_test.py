#!/usr/bin/env python3
"""
RepViT 검출 평가(Test) YOLO-style 런처.
반드시 detection/ 디렉토리에서 실행하세요.

사용 예시:
    python repvit_test.py \\
        --config     configs/faster_rcnn_repvit_m1_1_fpn_1x_qnsfl.py \\
        --checkpoint results/repvit_m1_1_qnsfl_baseline/latest.pth \\
        --device     1 \\
        --split      test \\
        --out        results/repvit_m1_1_qnsfl_baseline/test_results.pkl
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import mmcv
import torch
from mmcv import Config
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import (get_device, get_root_logger,
                         setup_multi_processes, update_data_root)

import repvit       # noqa: F401  backbone 등록
import repvit_mica  # noqa: F401  repvit_m1_1_mica 등록


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='RepViT Detection — YOLO-style test launcher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 필수
    p.add_argument('--config',     required=True,  help='mmdet config 파일 경로')
    p.add_argument('--checkpoint', required=True,  help='평가할 checkpoint (.pth) 경로')

    # 선택
    p.add_argument('--device',  type=str, default='0',
                   help='GPU 번호 (단일만 지원). 예: "1"')
    p.add_argument('--split',   type=str, default='test',
                   choices=['val', 'test'],
                   help='평가할 split (val 또는 test)')
    p.add_argument('--imgsz',   type=int, default=None,
                   help='입력 이미지 크기 오버라이드 (정사각형)')
    p.add_argument('--workers', type=int, default=4,
                   help='DataLoader 워커 수')
    p.add_argument('--out',     type=str, default=None,
                   help='결과 pkl 저장 경로 (생략 시 저장 안 함)')
    p.add_argument('--show-score-thr', type=float, default=0.3,
                   help='시각화 출력 최소 confidence 임계값')

    return p.parse_args()


def apply_test_overrides(cfg, args):
    """테스트 전용 오버라이드."""

    cfg.gpu_ids = [int(args.device)]

    if args.imgsz is not None:
        sz = args.imgsz
        for split in ('val', 'test'):
            for step in getattr(cfg.data, split).pipeline:
                if step.get('type') == 'MultiScaleFlipAug':
                    step['img_scale'] = (sz, sz)
                    break

    # 워커 수
    cfg.data.workers_per_gpu = args.workers

    return cfg


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    update_data_root(cfg)
    cfg = apply_test_overrides(cfg, args)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.device = get_device()
    gpu_id = cfg.gpu_ids[0]

    # ── Dataset ───────────────────────────────────────────────────────────────
    split_dataset_cfg = cfg.data.val if args.split == 'val' else cfg.data.test
    dataset = build_dataset(split_dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    wrap_fp16_model(model)

    print(f'\n  checkpoint : {args.checkpoint}')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # ── 평가 실행 ─────────────────────────────────────────────────────────────
    print(f'  split      : {args.split}')
    print(f'  images     : {len(dataset)}')
    print(f'  device     : {device}\n')

    outputs = single_gpu_test(model, data_loader, show=False)

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    if args.out:
        os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
        mmcv.dump(outputs, args.out)
        print(f'\n  결과 저장 완료: {args.out}')

    # ── mAP 계산 ──────────────────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('  COCO mAP 평가')
    print('─' * 60)
    eval_results = dataset.evaluate(outputs, metric='bbox',
                                    classwise=True,
                                    proposal_nums=(100, 300, 1000))

    print('\n' + '─' * 60)
    print('  클래스별 AP 요약')
    print('─' * 60)
    for k, v in eval_results.items():
        print(f'  {k:<30} : {v:.4f}')
    print('─' * 60)


if __name__ == '__main__':
    main()
