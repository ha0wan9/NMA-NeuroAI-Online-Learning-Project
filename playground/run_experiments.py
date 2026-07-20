# %% [markdown]
# Run Continual Learning Experiments (Predictive‑Coding MLP)
#
# This script/notebook imports data_utils.py and model_pcn.py produced earlier, runs training across tasks,
# computes accuracy and forgetting, and plots results.

# %%
# (Optional) install packages
# Uncomment if needed in the environment
# !pip install torch torchvision tqdm matplotlib jupytext

# %%
import os
import copy
import random
import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import build_split_mnist_loaders, build_permuted_mnist_loaders, build_rotated_mnist_loaders, evaluate_model_on_loader, plot_accuracy_matrix
from model_pcn import PCN_MLP

# %%
def run_experiment(experiment='split', num_tasks=5, batch_size=128, epochs_per_task=3,
                   hidden_size=400, lr=1e-3, lr_infer=0.1, infer_steps=20, lambda_sup=1.0,
                   activation='tanh', seed=0, use_cuda=False, download=True):
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    if experiment == 'split':
        train_loaders, test_loaders = build_split_mnist_loaders(batch_size=batch_size, num_tasks=num_tasks, seed=seed, download=download)
    elif experiment == 'perm':
        train_loaders, test_loaders = build_permuted_mnist_loaders(batch_size=batch_size, num_tasks=num_tasks, seed=seed, download=download)
    elif experiment == 'rotate':
        train_loaders, test_loaders = build_rotated_mnist_loaders(batch_size=batch_size, num_tasks=num_tasks, seed=seed, download=download)
    else:
        raise ValueError("Unknown experiment")

    input_dim = 28*28
    model = PCN_MLP(input_dim=input_dim,
                    hidden_sizes=[hidden_size, hidden_size],
                    output_dim=10,
                    activation=activation,
                    device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    accuracies = np.zeros((num_tasks, num_tasks), dtype=float)
    peak_acc_per_task = np.zeros(num_tasks, dtype=float)

    for t in range(num_tasks):
        print(f"Training task {t+1}/{num_tasks}")
        loader = train_loaders[t]
        for epoch in range(epochs_per_task):
            model.train()
            pbar = tqdm(loader, desc=f"Task {t+1} Epoch {epoch+1}", leave=False)
            for xb, yb in pbar:
                inferred = model.infer_hidden_states(xb, yb, n_steps=infer_steps, lr_infer=lr_infer, lambda_sup=lambda_sup)
                loss_val = model.outer_update_weights(xb, inferred, opt, lambda_sup=lambda_sup)
                pbar.set_postfix({'outer_loss': f"{loss_val:.4f}"})
        for k in range(t+1):
            acc = evaluate_model_on_loader(model, test_loaders[k], device=device)
            accuracies[t, k] = acc
            if acc > peak_acc_per_task[k]:
                peak_acc_per_task[k] = acc
        row = accuracies[t, :t+1]
        print(f"After task {t+1} accuracies: {[f'{x:.3f}' for x in row]}")
    final_accs = accuracies[-1, :].copy()
    avg_acc = final_accs.mean()
    forgetting_per_task = peak_acc_per_task - final_accs
    avg_forgetting = forgetting_per_task.mean()
    results = {
        'accuracies_matrix': accuracies,
        'peak_acc_per_task': peak_acc_per_task,
        'final_accs': final_accs,
        'avg_acc': avg_acc,
        'forgetting_per_task': forgetting_per_task,
        'avg_forgetting': avg_forgetting,
        'model_state': copy.deepcopy(model.state_dict())
    }
    return results

# %%
def run_and_plot(**kwargs):
    results = run_experiment(**kwargs)
    print("\nSummary:")
    for k in range(len(results['final_accs'])):
        print(f"Task {k}: peak {results['peak_acc_per_task'][k]:.3f}, final {results['final_accs'][k]:.3f}, forgetting {results['forgetting_per_task'][k]:.3f}")
    print(f"ACC (average final acc): {results['avg_acc']:.4f}")
    print(f"FORG (average forgetting): {results['avg_forgetting']:.4f}")
    plot_accuracy_matrix(results['accuracies_matrix'], title=f"Acc matrix ({kwargs.get('experiment','split')})")
    # Save summary
    fname = f"pcn_{kwargs.get('experiment','split')}_tasks{kwargs.get('num_tasks',5)}_summary.json"
    out = {
        'avg_acc': float(results['avg_acc']),
        'avg_forgetting': float(results['avg_forgetting']),
        'final_accs': [float(x) for x in results['final_accs'].tolist()],
        'peak_accs': [float(x) for x in results['peak_acc_per_task'].tolist()]
    }
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote summary to {fname}")
    return results

# %%
if __name__ == "__main__":
    # Quick demo (small settings for a fast run). Increase epochs/infer steps for real experiments.
    results = run_and_plot(experiment='split', num_tasks=5, batch_size=256, epochs_per_task=1,
                           hidden_size=200, lr=1e-3, lr_infer=0.1, infer_steps=10, lambda_sup=1.0,
                           activation='tanh', seed=0, use_cuda=False, download=True)
