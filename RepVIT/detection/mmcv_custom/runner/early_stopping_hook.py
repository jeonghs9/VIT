from mmcv.runner import HOOKS
from mmcv.runner.hooks import Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """val 지표가 patience epoch 동안 개선되지 않으면 학습을 조기 종료합니다."""

    def __init__(self, monitor='bbox_mAP', patience=5, rule='greater',
                 min_delta=0.001):
        assert rule in ('greater', 'less')
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.rule = rule
        self._best = None
        self._counter = 0

    def _is_improved(self, current):
        if self._best is None:
            return True
        if self.rule == 'greater':
            return current > self._best + self.min_delta
        return current < self._best - self.min_delta

    def after_val_epoch(self, runner):
        value = runner.log_buffer.output.get(self.monitor)
        if value is None:
            return

        if self._is_improved(value):
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            runner.logger.info(
                f'EarlyStoppingHook: {self.monitor}={value:.4f} '
                f'개선 없음 ({self._counter}/{self.patience})'
            )
            if self._counter >= self.patience:
                runner.logger.info(
                    f'EarlyStoppingHook: {self.patience} epoch 동안 개선 없음 — 학습 종료'
                )
                # runner.run() 루프 조건이 epoch < _max_epochs 이므로
                # _max_epochs 를 현재 epoch+1 로 줄여서 다음 루프에서 종료
                runner._max_epochs = runner.epoch + 1
