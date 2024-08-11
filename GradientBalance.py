import torch
import numpy as np
from torch.optim.optimizer import Optimizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GradientBalance(Optimizer):

    def __init__(self, params, relax_factor=0.1, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(GradientBalance, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array, nonshared_idx):
        self.balance_GradMagnitudes(loss_array, nonshared_idx)

    def balance_GradMagnitudes(self, loss_array, nonshared_idx):
        grad_task = []

        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p_idx, p in enumerate(group['params']):
                    if loss_index == 0:
                        grad_task.append(p.grad.detach().clone())

                    if p_idx >= nonshared_idx:
                        continue

                    if p.grad is None:
                        print("breaking")
                        break

                    if p.grad.is_sparse:
                        raise RuntimeError('HMG does not support sparse gradients')

                    if p.grad.equal(torch.zeros_like(p.grad)):
                        continue
                    
                    state = self.state[p] 

                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0:
                                p.norms = [torch.zeros(1).cuda()]
                            else:
                                p.norms.append(torch.zeros(1).cuda())

                    beta = group['beta']
                    p.norms[loss_index] = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))

                    relax_factor = group['relax_factor']

                    if p.norms[loss_index] > p.norms[0]:
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if inner_p < 0:
                            
                            p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]
                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)
                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']
            