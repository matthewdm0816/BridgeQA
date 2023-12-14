import torch

class MultiLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lambda_factories, last_epoch=-1, verbose=False):
        self.schedulers = []
        values = self._get_optimizer_lr(optimizer)
        for idx, factory in enumerate(lambda_factories):
            self.schedulers.append(factory(optimizer))
            values[idx] = self._get_optimizer_lr(optimizer)[idx]
            self._set_optimizer_lr(optimizer, values)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        result = []
        for idx, sched in enumerate(self.schedulers):
            result.append(sched.get_last_lr()[idx])
        return result
    
    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        for param_group, lr in zip(optimizer.param_groups, values):
            param_group['lr'] = lr

    @staticmethod
    def _get_optimizer_lr(optimizer):
        return [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for idx, sched in enumerate(self.schedulers):
                sched.step()
                values[idx] = self._get_optimizer_lr(self.optimizer)[idx]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()