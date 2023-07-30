import torch
from torch.optim.lr_scheduler import _LRScheduler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
Collected from https://github.com/cmpark0126/pytorch-polynomial-lr-decay
and customized as needed. 
And the original license from them are below as is: 
'
MIT License

Copyright (c) 2019 Park Chun Myong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'
"""


class PolynomialLRDecayWithOsc(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate, power=0.9, warm_up_epochs=0, decay_alpha=2.0,
                 decay_beta=0.5):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = -1  # First step is called before first epoch and it becomes 0
        assert 0 <= warm_up_epochs <= max_decay_steps
        assert decay_alpha >= 1
        assert decay_beta <= 1
        self.warm_up_epochs = warm_up_epochs
        self.decay_alpha = decay_alpha
        self.decay_beta = decay_beta
        super().__init__(optimizer)

    def get_lr(self):
        print('inside get_lr')
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]
        alpha_coeff = (self.decay_alpha + ((self.last_step - self.warm_up_epochs) % 2) * self.decay_beta)
        poly_coefficient = ((1 - max(0, (self.last_step - self.warm_up_epochs)) / (
                self.max_decay_steps * alpha_coeff)) ** self.power)

        return [(_base_lr - self.end_learning_rate) * poly_coefficient + self.end_learning_rate
                for _base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1 if self.last_step >= 0 else 0
        self.last_step = step  # if step != 0 else 1
        logger.debug('self.last_step:{}'.format(self.last_step))
        if self.last_step <= self.max_decay_steps:
            alpha_coeff = (self.decay_alpha + ((self.last_step - self.warm_up_epochs) % 2) * self.decay_beta)
            poly_coefficient = ((1 - max(0, (self.last_step - self.warm_up_epochs)) / (
                    self.max_decay_steps * alpha_coeff)) ** self.power)

            decay_lrs = [(_base_lr - self.end_learning_rate) * poly_coefficient + self.end_learning_rate
                         for _base_lr in self.base_lrs]
            # print('last step:%d alpha:%e pc:%e lr:%e'%(self.last_step, alpha_coeff, poly_coefficient, decay_lrs[0]))
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


class PolynomialLRDecayV2(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, total_epochs, max_decay_steps, end_learning_rate_factor=0.001, power=0.9):
        self.total_epochs = total_epochs
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate_factor = end_learning_rate_factor
        self.power = power
        self.last_step = -1  # First step is callled before first epoch and it becomes 0
        super().__init__(optimizer)

    def get_lr(self):
        print('inside get_lr')
        if self.last_step > self.max_decay_steps:
            return [base_lr * self.end_learning_rate_factor for base_lr in self.base_lrs]

        return [(base_lr - base_lr * self.end_learning_rate_factor) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                base_lr * self.end_learning_rate_factor
                for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1 if self.last_step >= 0 else 0
        self.last_step = step  # if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - base_lr * self.end_learning_rate_factor) *
                         ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                         base_lr * self.end_learning_rate_factor for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate, power=0.9):
        if max_decay_steps < 0.:
            raise ValueError('max_decay_steps should be greater than 0.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = -1  # First step is called before first epoch and it becomes 0
        super().__init__(optimizer)

    def get_lr(self):
        print('inside get_lr')
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                self.end_learning_rate
                for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1 if self.last_step >= 0 else 0
        self.last_step = step  # if step != 0 else 1
        logger.debug('self.last_step:{}'.format(self.last_step))
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr



def test_get_lrs(optim, scheduler, epochs):
    lr_0 = []
    # import ipdb; ipdb.set_trace()
    for epoch in range(0, epochs):
        lr_0.append(optim.param_groups[0]['lr'])
        scheduler.step()

    return lr_0


if __name__ == '__main__':
    base_lr = 1e-5
    epochs = 50

    v = torch.zeros(10)
    v2 = v.clone()
    param_dicts = [
        {'params': [v], 'lr': base_lr},
        {'params': [v2], 'lr': base_lr / 10}
    ]
    #optim = torch.optim.AdamW(param_dicts, lr=base_lr, weight_decay=0.9)
    #scheduler = PolynomialLRDecay(optim, max_decay_steps=epochs,
    #                              end_learning_rate=0.01 * base_lr, power=0.9)

    optim2 = torch.optim.AdamW(param_dicts, lr=base_lr, weight_decay=0.9)
    scheduler2 = PolynomialLRDecayWithOsc(optim2, max_decay_steps=50, end_learning_rate=0.01 * base_lr, power=0.9, warm_up_epochs=5, decay_alpha=3, decay_beta=0.2)

    #lr_vanilla = test_get_lrs(optim, scheduler, epochs)
    lr_alpha = test_get_lrs(optim=optim2, scheduler=scheduler2, epochs=epochs)
    # print(lr_vanilla)
    print(lr_alpha)
    # import ipdb; ipdb.set_trace()
    from matplotlib import pyplot as plt

    xx = range(epochs)
    # plt.plot(xx, lr_vanilla, label='vanilla')
    plt.plot(xx, lr_alpha, label='alpha')
    plt.show()
