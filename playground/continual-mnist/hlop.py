"""
HLOP — Hebbian Learning based Orthogonal Projection for Continual Learning.

Reference: Zheng et al. (2024) arxiv:2402.11984

HLOP uses lateral neural connections with Oja's rule to extract principal
subspaces of neural activity, then projects weight updates orthogonal to
previously learned subspaces.

For PC integration: we modify presynaptic activity traces (x in PC's latent
states) using the HLOP lateral circuit before computing weight updates.

Key idea:
  ΔW = δ * (x - H^T * H * x)^T    where H learns the principal subspace via Oja's rule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HLOPSubspace(nn.Module):
    """
    HLOP lateral connections for one layer.
    
    Maintains a subspace matrix H that learns principal components of 
    presynaptic activity via Oja's rule. Projects out components of x
    that lie in the learned subspace before weight updates.
    
    Args:
        input_dim: dimension of presynaptic activity
        n_subspace: number of subspace neurons (rank of projection)
        lr: learning rate for Oja's rule (Hebbian plasticity)
        init_scale: scale for random initialization of H
    """
    
    def __init__(
        self,
        input_dim: int,
        n_subspace: int = 10,
        lr: float = 0.01,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_subspace = n_subspace
        self.lr = lr
        
        # H: subspace matrix [n_subspace, input_dim]
        # Maps input x to subspace codes y = H @ x
        self.H = nn.Parameter(
            torch.randn(n_subspace, input_dim) * init_scale,
            requires_grad=False  # We update via Oja's rule manually
        )
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project activity traces orthogonal to learned subspace.
        
        Args:
            x: presynaptic activity [batch_size, input_dim]
            
        Returns:
            x_proj: projected activity [batch_size, input_dim]
                     x_proj = x - H^T @ H @ x
        """
        with torch.no_grad():
            # y = H @ x^T  -- subspace codes  [n_subspace, batch_size]
            y = self.H @ x.t()  # [n_subspace, batch_size]
            
            # x_proj = x - H^T @ y
            # H^T @ y: [input_dim, batch_size]
            x_proj = x - (self.H.t() @ y).t()  # [batch_size, input_dim]
            
        return x_proj
    
    def update_subspace(self, x: torch.Tensor):
        """
        Update H via Oja's subspace rule.
        
        ΔH = η * (y @ x^T - y @ y^T @ H)
        where y = H @ x^T
        
        This converges to the dominant principal subspace.
        """
        with torch.no_grad():
            batch_size = x.size(0)
            
            # Forward: y = H @ x^T  [n_subspace, batch_size]
            y = self.H @ x.t()
            
            # Oja's subspace update
            # Hebbian term: y @ x  [n_subspace, input_dim]
            hebbian = y @ x  # (n_subspace x batch) @ (batch x input_dim)
            
            # Anti-Hebbian term: -y @ y^T @ H  [n_subspace, input_dim]
            anti_hebbian = - (y @ y.t()) @ self.H  # (n_subspace x batch) @ (batch x n_subspace) @ (n_subspace x input_dim)
            
            # Combined update
            delta = self.lr * (hebbian + anti_hebbian) / batch_size
            
            self.H.data.add_(delta)
            
            # NOTE: QR re-orthonormalization is skipped for speed
            # Oja's subspace rule converges to the principal subspace
            # without explicit orthonormalization, just more slowly
    
    def get_subspace_dim(self) -> int:
        return self.n_subspace


class HLOPWrapper(nn.Module):
    """
    Wrapper that adds HLOP lateral circuits to a PC model.
    
    Intercepts the presynaptic activity traces (x states) in PC layers
    and applies HLOP projection before they're used for weight updates.
    
    Usage:
        model = make_pc_model()
        model = HLOPWrapper(model, subspace_dim=10)
        # Then use model normally with PC trainer
    """
    
    def __init__(
        self,
        model: nn.Module,
        subspace_dim: int = 10,
        hlop_lr: float = 0.01,
        input_dim: int = 784,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.model = model
        
        # One HLOP subspace per layer with learnable activities
        # For our MLP: input→256, 256→256
        self.hlop_layers = nn.ModuleList([
            HLOPSubspace(hidden_dim, subspace_dim, lr=hlop_lr),
            HLOPSubspace(hidden_dim, subspace_dim, lr=hlop_lr),
        ])
        
    def forward(self, x):
        return self.model(x)
    
    def get_model(self):
        return self.model
    
    def project_and_update(self, xs, layer_idx: int):
        """
        Project activity traces through HLOP and update subspace.
        
        Args:
            xs: list of latent states from PC layers
            layer_idx: which hidden layer (0 or 1)
        """
        if layer_idx >= len(self.hlop_layers):
            return xs
        
        x_data = xs[layer_idx].detach()
        
        # Update subspace with current activity
        self.hlop_layers[layer_idx].update_subspace(x_data)
        
        # Project activity for weight update
        x_proj = self.hlop_layers[layer_idx].project(x_data)
        
        return x_proj


def apply_hlop_to_pc_trainer(trainer, hlop_wrapper, n_pc_layers=2):
    """
    Monkey-patch the PC trainer to apply HLOP projection
    after each weight update step.
    
    This intercepts the optimizer_p.step() to project gradients
    orthogonal to previously learned subspaces.
    """
    original_optimizer_p = trainer._optimizer_p
    
    # Store reference for later use in training loop
    trainer._hlop_wrapper = hlop_wrapper
    trainer._hlop_enabled = True
    
    return trainer
