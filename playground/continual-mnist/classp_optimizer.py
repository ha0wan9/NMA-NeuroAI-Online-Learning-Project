"""Paper-equation CLASSP optimizer for continual learning.

Normative source: Ludwig (2024), arXiv:2405.09637, equation (1) and
Algorithm 1.  This implementation intentionally differs from the author's
reference ``CLASSP.py`` at commit ``ea3fe3c67279e27db8edea73d5f3ad522bb15dc1``:

* thresholding is elementwise, as equation (1) requires;
* the accumulator adds elementwise ``abs(gradient) ** p``;
* coordinates rejected by the threshold retain their prior accumulator; and
* ``apply_decay=False`` keeps accumulating history and applies the unscaled
  thresholded SGD update, matching the schedule described in section 4.1.

With ``p=2`` and ``threshold=0``, the update is AdaGrad when epsilon is
matched as an initial accumulator value (and PyTorch AdaGrad's additive
post-root epsilon is set to zero).
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional


class CLASSP(Optimizer):
    """
    CLASSP optimizer.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate
        threshold: gradient threshold for sparsity (only update if grad² > threshold)
        p: norm power for the decay term
        eps: small constant for numerical stability
        apply_decay: whether to apply the decay term (if False, acts as SGD with threshold)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        threshold: float = 0.0,
        p: float = 2.0,
        eps: float = 1e-8,
        apply_decay: bool = True,
    ):
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"Invalid threshold: {threshold}")
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError(f"Invalid p: {p}")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not isinstance(apply_decay, bool):
            raise TypeError("apply_decay must be a bool")

        defaults = dict(
            lr=lr, threshold=threshold, p=p, eps=eps, apply_decay=apply_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, *, apply_decay: Optional[bool] = None):
        """Perform one paper-equation optimization step.

        ``apply_decay`` may override the parameter-group setting for this
        step.  When false, the same threshold and accumulator updates apply,
        but the accepted coordinates use the unscaled learning rate.  This is
        the paper's first-task accumulation schedule, not a state reset.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if apply_decay is not None and not isinstance(apply_decay, bool):
            raise TypeError("apply_decay must be a bool or None")

        for group in self.param_groups:
            lr = group["lr"]
            threshold = group["threshold"]
            p_norm = group["p"]
            eps = group["eps"]
            use_decay = (
                group["apply_decay"] if apply_decay is None else apply_decay
            )

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if grad.is_sparse:
                    raise RuntimeError("CLASSP does not support sparse gradients")
                if not torch.isfinite(grad).all():
                    raise FloatingPointError("CLASSP received a non-finite gradient")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # grad_sum accumulates |grad|^p
                    state["grad_sum"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                grad_sum = state["grad_sum"]

                # Equation (1) is coordinate-wise and uses a strict inequality.
                mask = grad.square() > threshold
                if not bool(mask.any()):
                    continue

                accepted_grad = torch.where(mask, grad, torch.zeros_like(grad))
                grad_sum.add_(accepted_grad.abs().pow(p_norm))

                if use_decay:
                    denom = (grad_sum + eps).pow(1.0 / p_norm)
                    p.addcdiv_(accepted_grad, denom, value=-lr)
                else:
                    p.add_(accepted_grad, alpha=-lr)

        return loss


class CLASSPAdam(Optimizer):
    """
    CLASSP acting on Adam-style momentum rather than raw gradients.
    Uses gradient threshold sparsity with Adam's adaptive momentum.

    For PC integration: wraps PC's internal parameter optimizer.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        threshold: float = 0.0,
        classp_p: float = 2.0,
        amsgrad: bool = False,
    ):
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
        if not 0.0 <= threshold:
            raise ValueError(f"Invalid threshold: {threshold}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            threshold=threshold, amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("CLASSPAdam does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    state_steps.append(state["step"])

            beta1, beta2 = group["betas"]
            self._adam_with_threshold(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs if group["amsgrad"] else None,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                threshold=group["threshold"],
            )

            # Update state steps
            for s in state_steps:
                s += 1

        return loss

    @staticmethod
    def _adam_with_threshold(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
        state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps, threshold,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** (step + 1)
            bias_correction2 = 1 - beta2 ** (step + 1)

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Apply CLASSP threshold
            if threshold > 0.0:
                mask = grad.pow(2) > threshold
                if not mask.any():
                    continue
                grad = grad * mask

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad and max_exp_avg_sqs is not None:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2 ** 0.5)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

            step_size = lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)
