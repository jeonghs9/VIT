#!/usr/bin/env python3
"""
학습 시작 전 전체 파이프라인 사전 검증 스크립트.
반드시 detection/ 디렉토리에서 실행하세요.

검증 항목:
  [1] Config  : mmdet config 파일 파싱 가능 여부
  [2] Dataset : COCO JSON 존재 여부 + 이미지 경로 샘플 접근 가능 여부
  [3] Model   : backbone / neck / head 빌드 가능 여부
  [4] Weights : 사전학습 가중치 파일 존재 여부 + 로드 가능 여부
  [5] Pipeline: 실제 이미지 1장을 train_pipeline 으로 변환 가능 여부
  [6] Forward : dummy tensor 로 모델 forward pass 1회 성공 여부

사용 예시:
    python repvit_validate.py \\
        --config configs/faster_rcnn_repvit_m1_1_fpn_1x_qnsfl.py
"""

import argparse
import json
import os
import os.path as osp
import sys
import traceback

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

PASS = '  ✔'
FAIL = '  ✘'
WARN = '  ⚠'

# RepVIT detection/train.py 가 요구하는 버전
REQUIRED = {
    'mmcv'  : ('1.3.0', '1.8.0'),   # 1.3 ≤ x < 1.8
    'mmdet' : ('2.14.0', '2.99.0'),  # 2.x 계열
}


# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f'\n{"─" * 60}')
    print(f'  [{title}]')
    print(f'{"─" * 60}')


def ok(msg):
    print(f'{PASS}  {msg}')


def fail(msg):
    print(f'{FAIL}  {msg}')


def warn(msg):
    print(f'{WARN}  {msg}')


# ─────────────────────────────────────────────────────────────────────────────
#  [0] 환경(패키지 버전) 검사
# ─────────────────────────────────────────────────────────────────────────────

def check_environment():
    section('0  환경 (패키지 버전)')
    import importlib
    from packaging.version import Version

    env_ok = True

    for pkg, (vmin, vmax) in REQUIRED.items():
        try:
            import importlib.metadata as meta
            # mmcv 는 mmcv-full 이름으로도 설치될 수 있음
            candidates = [pkg, f'{pkg}-full'] if pkg == 'mmcv' else [pkg]
            ver = None
            for name in candidates:
                try:
                    ver = meta.version(name)
                    break
                except meta.PackageNotFoundError:
                    continue
            if ver is None:
                raise meta.PackageNotFoundError(pkg)
            v   = Version(ver)
            vlo = Version(vmin)
            vhi = Version(vmax)
            if vlo <= v < vhi:
                ok(f'{pkg} {ver}  (요구: >={vmin}, <{vmax})')
            else:
                fail(f'{pkg} {ver}  ← 버전 불일치!  요구: >={vmin}, <{vmax}')
                env_ok = False
        except Exception:
            fail(f'{pkg} 버전 확인 불가 (미설치 또는 충돌)')
            env_ok = False

    # torch / CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        ok(f'torch {torch.__version__}  CUDA available: {cuda_avail}')
        if cuda_avail:
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f'       GPU {i}: {name}  ({mem:.1f} GB)')
    except ImportError:
        fail('torch 미설치')
        env_ok = False

    if not env_ok:
        print()
        print('  ── 환경 수정 방법 ──────────────────────────────────────')
        print('  현재 시스템: mmcv 2.x + mmdet 3.x  (mmdet 3.x API)')
        print('  필요 버전  : mmcv 1.x + mmdet 2.x  (RepVIT detection 코드 기준)')
        print()
        print('  ① 기존 패키지 제거:')
        print('     pip uninstall mmcv mmdet mmengine -y')
        print()
        print('  ② mmcv-full 설치 (torch 1.10 + CUDA 10.2):')
        print('     pip install mmcv-full==1.7.1 \\')
        print('       -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html')
        print()
        print('  ③ mmdet 2.x 설치:')
        print('     pip install mmdet==2.28.2')
        print('  ────────────────────────────────────────────────────────')

    return env_ok


# ─────────────────────────────────────────────────────────────────────────────
#  [1] Config 파싱
# ─────────────────────────────────────────────────────────────────────────────

def check_config(config_path):
    section('1  Config 파싱')
    try:
        from mmcv import Config
        cfg = Config.fromfile(config_path)
        ok(f'Config 로드 성공: {config_path}')

        # 필수 키 존재 확인
        for key in ('model', 'data', 'runner', 'optimizer'):
            if hasattr(cfg, key):
                ok(f'필수 키 존재: {key}')
            else:
                fail(f'필수 키 없음: {key}')
                return None

        print(f'\n  epoch      : {cfg.runner.max_epochs}')
        print(f'  batch/gpu  : {cfg.data.samples_per_gpu}')
        print(f'  gpu_ids    : {cfg.gpu_ids}')
        print(f'  lr         : {cfg.optimizer.lr}')
        return cfg

    except Exception as e:
        fail(f'Config 로드 실패: {e}')
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  [2] Dataset / 이미지 경로
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset(cfg, n_sample=5):
    section('2  Dataset JSON + 이미지 경로')
    all_ok = True

    for split in ('train', 'val', 'test'):
        split_cfg = getattr(cfg.data, split, None)
        if split_cfg is None:
            warn(f'{split} split 설정 없음')
            continue

        ann_file = split_cfg.get('ann_file', '')
        if not osp.isfile(ann_file):
            fail(f'{split} JSON 없음: {ann_file}')
            all_ok = False
            continue

        try:
            with open(ann_file) as f:
                coco = json.load(f)
            n_img = len(coco.get('images', []))
            n_ann = len(coco.get('annotations', []))
            ok(f'{split} JSON 로드 성공 — images: {n_img:,}  annotations: {n_ann:,}')
        except Exception as e:
            fail(f'{split} JSON 파싱 오류: {e}')
            all_ok = False
            continue

        # 이미지 파일 샘플 접근 테스트
        images = coco.get('images', [])
        prefix = split_cfg.get('img_prefix', '')
        missing = 0
        for img in images[:n_sample]:
            full_path = osp.join(prefix, img['file_name']) if prefix else img['file_name']
            if not osp.isfile(full_path):
                missing += 1
                warn(f'  이미지 없음: {full_path}')
        if missing == 0:
            ok(f'{split} 이미지 샘플 {n_sample}장 경로 확인 완료')
        else:
            fail(f'{split} 샘플 {missing}/{n_sample}장 경로 오류')
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  [3] Model 빌드
# ─────────────────────────────────────────────────────────────────────────────

def check_model_build(cfg):
    section('3  Model 빌드')
    try:
        import repvit  # noqa: F401  backbone 등록
        from mmdet.models import build_detector
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'),
        )
        ok('모델 빌드 성공')

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f'  전체 파라미터 : {n_params:.2f} M')
        print(f'  학습 파라미터 : {n_train:.2f} M')
        return model

    except Exception as e:
        fail(f'모델 빌드 실패: {e}')
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  [4] 사전학습 가중치
# ─────────────────────────────────────────────────────────────────────────────

def check_weights(cfg):
    section('4  사전학습 가중치')
    try:
        ckpt_path = cfg.model.backbone.get('init_cfg', {}).get('checkpoint', None)
        if not ckpt_path:
            warn('backbone init_cfg.checkpoint 설정 없음 — scratch 학습')
            return True

        if not osp.isfile(ckpt_path):
            fail(f'가중치 파일 없음: {ckpt_path}')
            print(f'\n  다운로드 방법:')
            print(f'    wget -P pretrain \\')
            print(f'      https://github.com/THU-MIG/RepViT/releases/download/'
                  f'v1.0/repvit_m1_1_distill_300e.pth')
            return False

        import torch
        state = torch.load(ckpt_path, map_location='cpu')
        n_keys = len(state.get('model', state))
        ok(f'가중치 로드 성공: {ckpt_path}')
        print(f'  key 수: {n_keys}')
        return True

    except Exception as e:
        fail(f'가중치 검사 실패: {e}')
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  [5] Train Pipeline (실제 이미지 1장)
# ─────────────────────────────────────────────────────────────────────────────

def check_pipeline(cfg):
    section('5  Train Pipeline (실제 이미지 변환)')
    try:
        import json as _json
        from mmdet.datasets.pipelines import Compose

        ann_file = cfg.data.train.ann_file
        with open(ann_file) as f:
            coco = _json.load(f)

        img_info = coco['images'][0]
        prefix   = cfg.data.train.get('img_prefix', '')
        img_path = osp.join(prefix, img_info['file_name']) if prefix else img_info['file_name']

        if not osp.isfile(img_path):
            warn(f'파이프라인 테스트 이미지 없음: {img_path}  — 건너뜀')
            return True

        # 어노테이션 준비
        ann_map = {}
        for ann in coco['annotations']:
            ann_map.setdefault(ann['image_id'], []).append(ann)
        anns = ann_map.get(img_info['id'], [])

        import numpy as np
        gt_bboxes = np.array([[a['bbox'][0], a['bbox'][1],
                                a['bbox'][0] + a['bbox'][2],
                                a['bbox'][1] + a['bbox'][3]] for a in anns],
                              dtype=np.float32).reshape(-1, 4)
        gt_labels = np.array([a['category_id'] - 1 for a in anns], dtype=np.int64)

        img_info_with_filename = dict(img_info, filename=img_path)
        result = dict(
            filename=img_path,
            img_prefix='',
            img_info=img_info_with_filename,
            ann_info=dict(bboxes=gt_bboxes, labels=gt_labels),
            # pre_pipeline() 이 원래 채워주는 필드들 — 수동 dict 생성 시 필수
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[],
        )

        pipeline = Compose(cfg.data.train.pipeline)
        out = pipeline(result)

        img_tensor = out['img'].data
        ok(f'파이프라인 변환 성공')
        print(f'  입력 이미지  : {img_path}')
        print(f'  출력 tensor  : shape={tuple(img_tensor.shape)}, '
              f'dtype={img_tensor.dtype}')
        print(f'  bbox 수      : {len(gt_bboxes)}')
        return True

    except Exception as e:
        fail(f'파이프라인 변환 실패: {e}')
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  [6] Forward Pass (dummy)
# ─────────────────────────────────────────────────────────────────────────────

def check_forward(model, cfg):
    section('6  Forward Pass (dummy tensor)')
    try:
        import torch

        gpu_id = cfg.gpu_ids[0] if cfg.gpu_ids else 0
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        print(f'  디바이스 : {device}')

        model = model.to(device)
        model.eval()

        # train_pipeline 에서 img_scale 추출
        img_scale = (800, 800)
        for step in cfg.data.train.pipeline:
            if step.get('type') == 'Resize':
                sc = step.get('img_scale')
                if isinstance(sc, (list, tuple)) and len(sc) >= 1:
                    img_scale = sc[0] if isinstance(sc[0], (list, tuple)) else sc
                break
        h, w = (img_scale[1], img_scale[0]) if isinstance(img_scale, (list, tuple)) else (800, 800)

        dummy = torch.zeros(1, 3, h, w, device=device)
        img_meta = [[dict(
            ori_shape=(h, w, 3),
            img_shape=(h, w, 3),
            pad_shape=(h, w, 3),
            scale_factor=1.0,
            flip=False,
        )]]

        with torch.no_grad():
            result = model.forward_dummy(dummy)

        ok('Forward pass 성공')
        if isinstance(result, (list, tuple)):
            shapes = [tuple(r.shape) if hasattr(r, 'shape') else type(r).__name__
                      for r in result[:3]]
            print(f'  출력 shapes (앞 3개) : {shapes}')
        return True

    except Exception as e:
        fail(f'Forward pass 실패: {e}')
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='RepViT 학습 파이프라인 사전 검증')
    p.add_argument('--config', required=True, help='mmdet config 파일 경로')
    p.add_argument('--skip-forward', action='store_true',
                   help='Forward pass 검사 생략 (GPU 없는 환경)')
    return p.parse_args()


def main():
    args = parse_args()

    print('\n' + '═' * 60)
    print('  RepViT Validation — 학습 파이프라인 사전 검증')
    print('═' * 60)

    results = {}

    # [0] Environment
    try:
        from packaging.version import Version
        results['Environment'] = check_environment()
    except ImportError:
        warn('packaging 모듈 없음 — 버전 검사 생략  (pip install packaging)')
        results['Environment'] = None

    # [1] Config
    cfg = check_config(args.config)
    results['Config'] = cfg is not None

    if cfg is None:
        print('\n  Config 로드 실패로 이후 검사를 중단합니다.')
        _print_summary(results)
        sys.exit(1)

    # [2] Dataset
    results['Dataset'] = check_dataset(cfg)

    # [3] Model
    model = check_model_build(cfg)
    results['Model Build'] = model is not None

    # [4] Weights
    results['Pretrained Weights'] = check_weights(cfg)

    # [5] Pipeline
    results['Train Pipeline'] = check_pipeline(cfg)

    # [6] Forward
    if args.skip_forward or model is None:
        warn('Forward pass 검사 생략')
        results['Forward Pass'] = None
    else:
        results['Forward Pass'] = check_forward(model, cfg)

    _print_summary(results)

    failed = [k for k, v in results.items() if v is False]
    sys.exit(1 if failed else 0)


def _print_summary(results):
    print('\n' + '═' * 60)
    print('  검증 결과 요약')
    print('═' * 60)
    all_pass = True
    for name, passed in results.items():
        if passed is True:
            status = '✔ PASS'
        elif passed is False:
            status = '✘ FAIL'
            all_pass = False
        else:
            status = '─ SKIP'
        print(f'  {status}  {name}')
    print('═' * 60)
    if all_pass:
        print('  ✔  모든 검증 통과 — 학습을 시작할 수 있습니다.\n')
    else:
        print('  ✘  실패 항목을 수정한 뒤 다시 검증하세요.\n')


if __name__ == '__main__':
    main()
