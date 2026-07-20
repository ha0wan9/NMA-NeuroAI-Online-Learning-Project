# %% [markdown]
# Data and Utilities for Continual MNIST Experiments
#
# This notebook builds dataset loaders for Split‑MNIST, Permuted‑MNIST, and Rotated‑MNIST,
# plus utility functions (evaluation, plotting). The notebook will also save a Python module
# `data_utils.py` for reuse by other notebooks.

# %%
# (Optional) Install dependencies if needed
# Uncomment if packages are missing in your environment
# !pip install torch torchvision tqdm matplotlib jupytext

# %%
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# %%
def get_split_mnist_tasks(num_tasks=5):
    assert num_tasks == 5, "Split-MNIST commonly uses 5 tasks of 2 classes each."
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    return pairs[:num_tasks]

def build_split_mnist_loaders(batch_size=128, num_tasks=5, train=True, download=True, seed=0):
    pairs = get_split_mnist_tasks(num_tasks)
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    full_test  = datasets.MNIST(root='./data', train=False, download=download, transform=transform)
    train_loaders = []
    test_loaders = []
    for (a,b) in pairs:
        def make_loader(dataset, bs, shuffle):
            idxs = [i for i, (_, label) in enumerate(dataset) if label in (a,b)]
            subset = Subset(dataset, idxs)
            return DataLoader(subset, batch_size=bs, shuffle=shuffle)
        train_loaders.append(make_loader(full_train, batch_size, True))
        test_loaders.append(make_loader(full_test, batch_size, False))
    return train_loaders, test_loaders

def build_permuted_mnist_loaders(batch_size=128, num_tasks=5, download=True, seed=0):
    base_transform = transforms.ToTensor()
    train = datasets.MNIST(root='./data', train=True, download=download, transform=base_transform)
    test  = datasets.MNIST(root='./data', train=False, download=download, transform=base_transform)
    imgs_train = torch.stack([x for x,_ in train])
    ys_train   = torch.tensor([y for _,y in train], dtype=torch.long)
    imgs_test  = torch.stack([x for x,_ in test])
    ys_test    = torch.tensor([y for _,y in test], dtype=torch.long)
    rng = np.random.RandomState(seed)
    train_loaders = []
    test_loaders = []
    flat_train = imgs_train.view(len(imgs_train), -1)
    flat_test  = imgs_test.view(len(imgs_test), -1)
    for t in range(num_tasks):
        perm = rng.permutation(flat_train.shape[1])
        perm_train = flat_train[:, perm].view(-1,1,28,28)
        perm_test  = flat_test[:, perm].view(-1,1,28,28)
        train_dataset = TensorDataset(perm_train, ys_train)
        test_dataset  = TensorDataset(perm_test, ys_test)
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders

def build_rotated_mnist_loaders(batch_size=128, num_tasks=5, download=True, seed=0):
    angles = np.linspace(0, 90, num_tasks, endpoint=True)
    base = datasets.MNIST(root='./data', train=True, download=download, transform=transforms.ToTensor())
    base_test = datasets.MNIST(root='./data', train=False, download=download, transform=transforms.ToTensor())
    train_loaders=[]
    test_loaders=[]
    for ang in angles:
        def rotate_dataset(dataset, angle):
            imgs = []
            labels = []
            for x,y in dataset:
                pil = TF.to_pil_image(x)
                rot = TF.rotate(pil, angle)
                xi = TF.to_tensor(rot)
                imgs.append(xi)
                labels.append(y)
            imgs = torch.stack(imgs)
            labels = torch.tensor(labels, dtype=torch.long)
            return TensorDataset(imgs, labels)
        train_d = rotate_dataset(base, float(ang))
        test_d = rotate_dataset(base_test, float(ang))
        train_loaders.append(DataLoader(train_d, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_d, batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders

# %%
def evaluate_model_on_loader(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model.forward(xb)
            preds = torch.argmax(logits, dim=1).cpu()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total>0 else 0.0

def plot_accuracy_matrix(acc_matrix, title='Accuracies'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.imshow(acc_matrix, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(label='Accuracy')
    plt.xlabel('Task evaluated')
    plt.ylabel('After training task')
    plt.title(title)
    plt.show()

# %%
# Save the helpers into a module file (optional)
if __name__ == "__main__":
    import inspect
    module_code = ''
    module_code += inspect.getsource(get_split_mnist_tasks) + '\n\n'
    module_code += inspect.getsource(build_split_mnist_loaders) + '\n\n'
    module_code += inspect.getsource(build_permuted_mnist_loaders) + '\n\n'
    module_code += inspect.getsource(build_rotated_mnist_loaders) + '\n\n'
    module_code += inspect.getsource(evaluate_model_on_loader) + '\n\n'
    module_code += inspect.getsource(plot_accuracy_matrix) + '\n\n'
    with open('data_utils.py', 'w') as f:
        f.write('import numpy as np\nimport torch\nfrom torch.utils.data import DataLoader, Subset, TensorDataset\nfrom torchvision import datasets, transforms\nimport torchvision.transforms.functional as TF\n\n')
        f.write(module_code)
    print("Wrote data_utils.py")
