"""Exact legacy-v0 CLASSP implementation retained for artifact reproduction.

This module intentionally preserves the July 21 implementation defect: when
thresholding is enabled, accumulator history is cleared on coordinates that
pass the threshold before the current contribution is added.  It must not be
used for new scientific claims.  The corrected paper-equation implementation
is in ``classp_optimizer.py``.
"""

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


class CLASSP(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        threshold: float = 0.0,
        p: float = 2.0,
        eps: float = 1e-8,
        apply_decay: bool = True,
    ):
        defaults = dict(
            lr=lr,
            threshold=threshold,
            p=p,
            eps=eps,
            apply_decay=apply_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("legacy CLASSP does not support sparse gradients")
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["grad_sum"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                state["step"] += 1
                grad_sum = state["grad_sum"]
                if group["threshold"] > 0.0:
                    mask = gradient.pow(2) > group["threshold"]
                    if not mask.any():
                        continue
                    gradient = gradient * mask
                    if group["apply_decay"]:
                        # Intentional legacy-v0 defect: active-coordinate
                        # history is erased at every accepted update.
                        grad_sum.mul_(~mask)
                        grad_sum.add_(gradient.abs().pow(group["p"]))
                elif group["apply_decay"]:
                    grad_sum.add_(gradient.abs().pow(group["p"]))
                if group["apply_decay"]:
                    denominator = (grad_sum + group["eps"]).pow(1.0 / group["p"])
                    denominator = denominator.clamp(min=group["eps"])
                    step_size = group["lr"] / denominator
                else:
                    step_size = group["lr"]
                parameter.add_(gradient * -step_size)
        return loss
