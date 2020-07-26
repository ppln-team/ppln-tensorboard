from torch.utils.tensorboard import SummaryWriter

from ppln.hooks import BaseHook
from ppln.hooks.priority import Priority
from ppln.hooks.registry import HOOKS
from ppln.utils.dist import master_only


@HOOKS.register_module
class TensorboardLoggerHook(BaseHook):
    def __init__(self, log_dir=None):
        super(TensorboardLoggerHook, self).__init__()
        self.log_dir = log_dir
        self.writer = None

    @property
    def priority(self):
        return Priority.VERY_LOW

    @master_only
    def before_run(self, runner):
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        for name, value in runner.epoch_outputs.items():
            if name in ["time", "data_time"]:
                continue
            tag = f"{name}/{runner.mode}"
            if isinstance(value, str):
                self.writer.add_text(tag, value, runner.iter)
            else:
                self.writer.add_scalar(tag, value, runner.iter)

    @master_only
    def after_run(self, runner):
        self.writer.close()

    def after_epoch(self, runner):
        self.log(runner)
