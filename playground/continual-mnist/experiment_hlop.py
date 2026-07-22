#!/usr/bin/env python3
"""
PC+HLOP Experiment — Hebbian Orthogonal Projection for Predictive Coding.

Reference: Zheng et al. (2024) arxiv:2402.11984

HLOP uses lateral neural circuits with Oja's rule to learn principal
subspaces of neural activity. Weight updates are projected orthogonal
to these subspaces, preventing catastrophic forgetting.

Integration with PC:
  1. After each task, extract PC latent states → update HLOP subspace
  2. During next task, project weight gradients orthogonal to subspace
  3. Only update weights in directions not already used by previous tasks
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import trange, tqdm

# Add paths
PC_LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'predictive-coding')
sys.path.insert(0, PC_LIB_PATH)
sys.path.insert(0, os.path.dirname(__file__))
import predictive_coding as pc
from hlop import HLOPSubspace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

def get_permuted_mnist(n_tasks=20, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    mnist_train = datasets.MNIST('./data', download=True, train=True, transform=transform)
    mnist_test = datasets.MNIST('./data', download=True, train=False, transform=transform)

    gen = torch.Generator().manual_seed(seed)
    permutations = [torch.arange(28*28)] + [
        torch.randperm(28*28, generator=gen) for _ in range(n_tasks - 1)
    ]

    class PermutedMNIST(torch.utils.data.Dataset):
        def __init__(self, dataset, permutation):
            self.dataset = dataset
            self.permutation = permutation
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            image, label = self.dataset[index]
            return image[self.permutation], label

    train_tasks = [PermutedMNIST(mnist_train, p) for p in permutations]
    test_tasks = [PermutedMNIST(mnist_test, p) for p in permutations]
    return train_tasks, test_tasks

# ---------------------------------------------------------------------------
# PC Model with HLOP hooks
# ---------------------------------------------------------------------------
def make_pc_model(hidden_size=HIDDEN_SIZE):
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden_size),
        pc.PCLayer(),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        pc.PCLayer(),
        nn.ReLU(),
        nn.Linear(hidden_size, OUTPUT_SIZE),
    )
    model.train()
    return model.to(device)

LOSS_FN = lambda output, _target: 0.5 * (output - _target).pow(2).sum()

@torch.no_grad()
def evaluate(model, dataset, batch_size=1000):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        if isinstance(output, (list, tuple)):
            output = output[0]
        pred = output.argmax(dim=-1)
        total += label.size(0)
        correct += (pred == label).sum().item()
    model.train()
    return correct / total

# ---------------------------------------------------------------------------
# HLOP Gradient Projection
# ---------------------------------------------------------------------------
def apply_hlop_projection(model, hlop_layers):
    """
    After backward pass, modify weight gradients using HLOP projection.
    
    For each weight W_k with presynaptic activity x:
      grad_W_HLOP = grad_W - grad_W @ H^T @ H
      
    where H is the subspace matrix learned by Oja's rule.
    """
    with torch.no_grad():
        # Find linear layers and their corresponding HLOP projections
        linear_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                if linear_idx < len(hlop_layers):
                    hlop = hlop_layers[linear_idx]
                    w_grad = module.weight.grad  # [out_dim, in_dim]
                    
                    # Project: new_grad = grad - grad @ H^T @ H
                    # H: [n_subspace, in_dim]
                    # H^T @ H: [in_dim, in_dim]
                    # grad @ H^T @ H: [out_dim, in_dim]
                    H = hlop.H  # [n_sub, in_dim]
                    HTH = H.t() @ H  # [in_dim, in_dim]
                    proj_contrib = w_grad @ HTH  # [out_dim, in_dim]
                    
                    module.weight.grad.data.sub_(proj_contrib)
                    
                linear_idx += 1


def extract_pc_latent_states(model, loader, n_batches=5):
    """
    Extract PC latent states (x values from PCLayers) from the model.
    Used to update HLOP subspaces after each task.
    """
    model.eval()
    xs_list = [[], []]
    
    for i, (data, label) in enumerate(loader):
        if i >= n_batches:
            break
        data, label = data.to(device), label.to(device)
        
        # Forward pass through PC layers to get latent states
        # Each PCLayer holds _x (the latent state)
        layer_idx = 0
        for module in model.modules():
            if isinstance(module, pc.PCLayer):
                if module.get_x() is not None:
                    xs_list[layer_idx].append(module.get_x().detach().cpu())
                layer_idx += 1
        
        if layer_idx == 0:
            # If no PC layers found, try direct forward pass
            _ = model(data)
            layer_idx = 0
            for module in model.modules():
                if isinstance(module, pc.PCLayer):
                    if module.get_x() is not None:
                        xs_list[layer_idx].append(module.get_x().detach().cpu())
                    layer_idx += 1
    
    model.train()
    
    # Concatenate
    result = []
    for xs in xs_list:
        if xs:
            result.append(torch.cat(xs, dim=0))
    return result


def update_hlop_subspaces(hlop_layers, xs_list):
    """
    Update HLOP subspaces using extracted PC latent states.
    """
    for i, xs in enumerate(xs_list):
        if i < len(hlop_layers) and xs is not None and xs.size(0) > 0:
            hlop_layers[i].update_subspace(xs.to(device))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_hlop_experiment(train_tasks, test_tasks, 
                        n_subspace=10, hlop_lr=0.01,
                        lr=0.001, T=20, lr_x=0.01,
                        batch_size=500, seed=42):
    """Run PC+HLOP continual learning experiment."""
    n_tasks = len(train_tasks)
    torch.manual_seed(seed)
    
    # Create model and PC trainer
    model = make_pc_model()
    p_opt = optim.Adam(model.parameters(), lr=lr)
    trainer = pc.PCTrainer(
        model, T=T,
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs={'lr': lr_x},
        update_x_at='all', update_p_at='last',
        manual_optimizer_p_fn=lambda: p_opt,
    )
    
    # Create HLOP subspaces for each hidden layer
    hlop_layers = nn.ModuleList([
        HLOPSubspace(HIDDEN_SIZE, n_subspace, lr=hlop_lr),
        HLOPSubspace(HIDDEN_SIZE, n_subspace, lr=hlop_lr),
    ]).to(device)
    
    # Accuracy matrix
    acc_matrix = np.zeros((n_tasks + 1, n_tasks))
    for j in range(n_tasks):
        acc_matrix[0, j] = evaluate(model, test_tasks[j])
    
    # Train sequentially
    for task_id in range(n_tasks):
        loader = torch.utils.data.DataLoader(
            train_tasks[task_id], batch_size=batch_size, shuffle=True
        )
        
        desc = f"PC+HLOP task {task_id+1}/{n_tasks}"
        for _ in trange(1, desc=desc, leave=False):
            for data, label in loader:
                data, label = data.to(device), label.to(device)
                target = F.one_hot(label, num_classes=OUTPUT_SIZE).float()
                
                # PC forward + backward
                trainer.train_on_batch(
                    inputs=data,
                    loss_fn=LOSS_FN,
                    loss_fn_kwargs={'_target': target},
                    is_reset_optimizer_x_at_batch_start=True,
                    is_reset_optimizer_p_at_batch_start=False,
                )
                
                # Apply HLOP gradient projection (after backward, before optimizer_p.step)
                if task_id > 0:  # No projection for first task
                    apply_hlop_projection(model, hlop_layers)
        
        # After task: extract latent states and update HLOP subspace
        if task_id > 0:
            eval_loader = torch.utils.data.DataLoader(
                train_tasks[task_id], batch_size=batch_size, shuffle=True
            )
            xs_list = extract_pc_latent_states(model, eval_loader, n_batches=3)
            update_hlop_subspaces(hlop_layers, xs_list)
        
        # Evaluate
        for j in range(n_tasks):
            acc_matrix[task_id + 1, j] = evaluate(model, test_tasks[j])
    
    # Metrics
    final_accs = acc_matrix[-1]
    forgetting = []
    for j in range(n_tasks):
        if j == 0:
            forgetting.append(0.0)
        else:
            peak = np.max(acc_matrix[1:j+2, j])
            forgetting.append(float(peak - final_accs[j]))
    
    results = {
        'config': {
            'method': 'pc_hlop',
            'n_subspace': n_subspace,
            'hlop_lr': hlop_lr,
            'n_tasks': n_tasks,
            'lr': lr, 'T': T, 'lr_x': lr_x,
            'batch_size': batch_size, 'seed': seed,
            'hidden_size': HIDDEN_SIZE,
        },
        'acc_matrix': acc_matrix.tolist(),
        'final_avg_acc': float(np.mean(final_accs)),
        'avg_forgetting': float(np.mean(forgetting[1:])) if n_tasks > 1 else 0.0,
        'avg_backward_transfer': float(np.mean([
            acc_matrix[-1, j] - acc_matrix[j+1, j]
            for j in range(n_tasks - 1)
        ])) if n_tasks > 1 else 0.0,
        'device': str(device),
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='PC+HLOP experiment')
    parser.add_argument('--n-tasks', type=int, default=20)
    parser.add_argument('--n-subspace', type=int, default=10,
                       help='Number of subspace neurons (rank of projection)')
    parser.add_argument('--hlop-lr', type=float, default=0.01,
                       help='Learning rate for Oja Hebbian update')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-x', type=float, default=0.01)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"PC+HLOP: n_tasks={args.n_tasks}, subspace={args.n_subspace}, hlop_lr={args.hlop_lr}")
    print(f"{'='*60}\n")
    
    train_tasks, test_tasks = get_permuted_mnist(n_tasks=args.n_tasks, seed=args.seed)
    
    t0 = time.time()
    results = run_hlop_experiment(
        train_tasks, test_tasks,
        n_subspace=args.n_subspace,
        hlop_lr=args.hlop_lr,
        lr=args.lr, T=args.T, lr_x=args.lr_x,
        batch_size=args.batch_size, seed=args.seed,
    )
    elapsed = time.time() - t0
    results['elapsed_seconds'] = elapsed
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Final avg accuracy: {results['final_avg_acc']:.4f}")
    print(f"  Avg forgetting:     {results['avg_forgetting']:.4f}")
    print(f"  Avg BWT:           {results['avg_backward_transfer']:.4f}")
    print(f"  Time:              {elapsed:.1f}s")
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory:   {max_mem:.2f} GB")
    
    for j in range(args.n_tasks):
        print(f"  Task {j:2d}: {results['acc_matrix'][-1][j]:.4f}")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
