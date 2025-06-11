import torch
import math
from torch.optim import Optimizer

class Sophie(Optimizer):
    """
    Реализация оптимизатора Sophie.
    
    Sophie - это адаптивный оптимизатор, который комбинирует преимущества Adam и RMSprop
    с дополнительной адаптацией к геометрии пространства параметров.
    
    Args:
        params: Итерируемый объект параметров для оптимизации
        lr: Скорость обучения (default: 1e-3)
        betas: Коэффициенты для вычисления скользящих средних градиента и его квадрата (default: (0.9, 0.999))
        eps: Терминал для численной стабильности (default: 1e-8)
        weight_decay: Коэффициент регуляризации L2 (default: 0.01)
        amsgrad: Использовать ли версию AMSGrad алгоритма (default: True)
        momentum: Коэффициент импульса (default: 0.9)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=True, momentum=0.9):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, momentum=momentum)
        super(Sophie, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Sophie, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', True)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Выполняет один шаг оптимизации.
        
        Args:
            closure: Замыкание, которое пересчитывает модель и возвращает loss
            
        Returns:
            loss: Значение функции потерь, если closure не None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Sophie не поддерживает разреженные градиенты')
                    
                state = self.state[p]

                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    # Экспоненциальное скользящее среднее градиента
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Экспоненциальное скользящее среднее квадрата градиента
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Импульс
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Сохраняем максимальное значение exp_avg_sq
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                momentum = group['momentum']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Применяем L2 регуляризацию
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Обновляем скользящие средние
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    # Обновляем максимальное значение exp_avg_sq
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Используем максимальное значение для нормализации
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Вычисляем адаптивный шаг
                step_size = group['lr'] / bias_correction1

                # Применяем импульс
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(momentum).add_(exp_avg / denom, alpha=step_size)
                    p.add_(momentum_buffer, alpha=-1)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss 