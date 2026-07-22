#!/usr/bin/env python3
"""
Legacy-v0 Continual MNIST Experiment Harness.

This file is retained to reproduce the July 21 exploratory artifacts. It
recreates parameter optimizers at task boundaries and does not implement the
corrected matched-lr-paper-v1 protocol. Do not use it for cross-task CLASSP or
PC×CLASSP claims; use run_matched_lr_validation.py instead.

Experiments:
  1. Vanilla PC vs Vanilla BP (baselines)
  2. PC+CLASSP vs BP+CLASSP
  3. iPC scheduling (update_p_at='all' vs 'last')

Usage:
  python3 experiment.py --method pc --n-tasks 20
  python3 experiment.py --method bp --classp --classp-p 2 --classp-threshold 1e-5
  python3 experiment.py --method pc --ipc --n-tasks 50
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import trange

PC_LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'predictive-coding')
sys.path.insert(0, PC_LIB_PATH)
sys.path.insert(0, os.path.dirname(__file__))
import predictive_coding as pc
from legacy_classp_optimizer import CLASSP

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    gname = torch.cuda.get_device_name(0)
    gmem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: {gname} — {gmem:.1f} GB")
else:
    print("Device: CPU")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
INPUT_SIZE = 28 * 28
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
# Models
# ---------------------------------------------------------------------------
ACTIVATION = nn.ReLU

def make_pc_model(hidden_size=256):
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden_size), pc.PCLayer(), ACTIVATION(),
        nn.Linear(hidden_size, hidden_size), pc.PCLayer(), ACTIVATION(),
        nn.Linear(hidden_size, OUTPUT_SIZE),
    )
    model.train()
    return model.to(device)

def make_bp_model(hidden_size=256):
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden_size), ACTIVATION(),
        nn.Linear(hidden_size, hidden_size), ACTIVATION(),
        nn.Linear(hidden_size, OUTPUT_SIZE),
    )
    model.train()
    return model.to(device)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
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
        total += label.size(0)
        correct += (output.argmax(dim=-1) == label).sum().item()
    model.train()
    return correct / total

LOSS_FN = lambda output, _target: 0.5 * (output - _target).pow(2).sum()

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_experiment(
    train_tasks, test_tasks,
    method='pc',
    epochs_per_task=1,
    lr=0.001,
    T=20,
    ipc=False,
    classp=False,
    classp_p=2.0,
    classp_threshold=0.0,
    lr_x=0.01,
    bp_sgd=False,
    hidden_size=256,
    seed=42,
    batch_size=500,
):
    """Run continual learning experiment. Returns results dict."""
    n_tasks = len(train_tasks)
    torch.manual_seed(seed)

    # --- Model ---
    model = make_pc_model(hidden_size) if method == 'pc' else make_bp_model(hidden_size)

    # --- Pre-training evaluation on all tasks ---
    initial_accs = [evaluate(model, test_tasks[j]) for j in range(n_tasks)]
    task0_curve = [initial_accs[0]]
    
    # Track peak accuracy for each task (right after it's trained)
    peak_accs = list(initial_accs)  # start with initial values

    # --- Training loop ---
    for task_id in range(n_tasks):
        loader = torch.utils.data.DataLoader(
            train_tasks[task_id], batch_size=batch_size, shuffle=True
        )

        tag = f"{method.upper()}"
        if ipc: tag += "+iPC"
        if classp: tag += "+CLASSP"
        desc = f"{tag} t{task_id+1}/{n_tasks}"

        if method == 'pc':
            opt = CLASSP(model.parameters(), lr=lr, p=classp_p,
                         threshold=classp_threshold) if classp else optim.Adam(model.parameters(), lr=lr)
            trainer = pc.PCTrainer(
                model, T=T,
                optimizer_x_fn=optim.SGD, optimizer_x_kwargs={'lr': lr_x},
                update_x_at='all', update_p_at='all' if ipc else 'last',
                manual_optimizer_p_fn=lambda: opt,
            )
            for _ in trange(epochs_per_task, desc=desc, leave=False):
                for data, label in loader:
                    data, label = data.to(device), label.to(device)
                    target = F.one_hot(label, num_classes=OUTPUT_SIZE).float()
                    trainer.train_on_batch(
                        inputs=data, loss_fn=LOSS_FN,
                        loss_fn_kwargs={'_target': target},
                        is_reset_optimizer_x_at_batch_start=True,
                        is_reset_optimizer_p_at_batch_start=False,
                    )
        else:  # bp
            if classp:
                opt = CLASSP(model.parameters(), lr=lr, p=classp_p,
                             threshold=classp_threshold)
            elif bp_sgd:
                opt = optim.SGD(model.parameters(), lr=lr)
            else:
                opt = optim.Adam(model.parameters(), lr=lr)

            ce = nn.CrossEntropyLoss()
            for _ in trange(epochs_per_task, desc=desc, leave=False):
                for data, label in loader:
                    data, label = data.to(device), label.to(device)
                    opt.zero_grad()
                    ce(model(data), label).backward()
                    opt.step()

        # Track task 0 (forgetting after each new task)
        task0_curve.append(evaluate(model, test_tasks[0]))
        # Track current task peak
        peak_accs[task_id] = evaluate(model, test_tasks[task_id])

    # --- Final evaluation on all tasks ---
    print("  Final evaluation...")
    final_accs = [evaluate(model, test_tasks[j]) for j in range(n_tasks)]

    # --- Metrics ---
    # Forgetting: peak accuracy on task j minus final accuracy on task j
    forgetting = [float(peak_accs[j] - final_accs[j]) for j in range(n_tasks)]
    # Backward transfer: final accuracy on task j minus initial accuracy on task j
    bwt = [float(final_accs[j] - initial_accs[j]) for j in range(n_tasks)]

    results = {
        'config': {
            'method': method, 'ipc': ipc, 'classp': classp,
            'classp_p': classp_p, 'classp_threshold': classp_threshold,
            'n_tasks': n_tasks, 'epochs_per_task': epochs_per_task,
            'lr': lr, 'T': T, 'lr_x': lr_x,
            'bp_sgd': bp_sgd, 'batch_size': batch_size,
            'seed': seed, 'hidden_size': hidden_size,
        },
        'initial_accs': initial_accs,
        'final_accs': final_accs,
        'peak_accs': peak_accs,
        'task0_curve': task0_curve,
        'final_avg_acc': float(np.mean(final_accs)),
        'avg_forgetting': float(np.mean(forgetting[1:])),  # exclude task 0
        'avg_bwt': float(np.mean(bwt)),
        'forgetting': forgetting,
        'device': str(device),
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Continual MNIST')
    p.add_argument('--method', choices=['pc', 'bp'], default='pc')
    p.add_argument('--n-tasks', type=int, default=20)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lr-x', type=float, default=0.01)
    p.add_argument('--T', type=int, default=20)
    p.add_argument('--ipc', action='store_true')
    p.add_argument('--classp', action='store_true')
    p.add_argument('--classp-p', type=float, default=2.0)
    p.add_argument('--classp-threshold', type=float, default=0.0)
    p.add_argument('--bp-sgd', action='store_true')
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"Experiment: {args.method.upper()}", end='')
    if args.ipc: print(f" + iPC", end='')
    if args.classp: print(f" + CLASSP(p={args.classp_p},θ={args.classp_threshold})", end='')
    print(f"\n  {args.n_tasks} tasks, {args.epochs} epoch/task, lr={args.lr}, bs={args.batch_size}")
    if args.method == 'pc': print(f"  T={args.T}, lr_x={args.lr_x}")
    print('='*60)

    train, test = get_permuted_mnist(n_tasks=args.n_tasks, seed=args.seed)

    t0 = time.time()
    r = run_experiment(train, test,
        method=args.method, epochs_per_task=args.epochs, lr=args.lr,
        T=args.T, ipc=args.ipc,
        classp=args.classp, classp_p=args.classp_p,
        classp_threshold=args.classp_threshold, lr_x=args.lr_x,
        bp_sgd=args.bp_sgd, hidden_size=args.hidden_size,
        batch_size=args.batch_size, seed=args.seed)

    r['elapsed_seconds'] = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Final avg acc:  {r['final_avg_acc']:.4f}")
    print(f"  Avg forgetting: {r['avg_forgetting']:.4f}")
    print(f"  Avg BWT:       {r['avg_bwt']:.4f}")
    print(f"  Task 0 curve:  {', '.join(f'{v:.3f}' for v in r['task0_curve'])}")
    print(f"  Time:          {r['elapsed_seconds']:.0f}s")
    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU mem:  {mem:.2f} GB")
        r['peak_gpu_memory_gb'] = mem
    print('='*60)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(r, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
