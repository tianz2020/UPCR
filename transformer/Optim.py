import ipdb


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step(self):
        "Step with the inner optimizer"
        # ipdb.set_trace()
        # for para in self._optimizer.param_groups[0]['params']:
        #     print(para)

        self._update_learning_rate()
        self._optimizer.step()

    def update_step(self, global_step):
        self.n_steps = global_step

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        self._optimizer.param_groups[0]['lr'] = lr

        if len(self._optimizer.param_groups)>1:
            self._optimizer.param_groups[1]['lr'] = 0.6*lr