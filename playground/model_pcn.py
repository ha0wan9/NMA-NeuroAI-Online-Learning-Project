# %% [markdown]
# Predictive‑Coding MLP (PCN_MLP) Module
#
# Defines the PCN_MLP class and saves it as `model_pcn.py` for reuse.

# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class PCN_MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_sizes=[400,400], output_dim=10, activation='tanh', device='cpu'):
        super().__init__()
        self.device = device
        dims = [input_dim] + hidden_sizes + [output_dim]
        self.n_layers = len(dims)-1
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i in range(self.n_layers):
            W = nn.Parameter(torch.randn(dims[i+1], dims[i]) * math.sqrt(2.0/(dims[i]+dims[i+1])))
            b = nn.Parameter(torch.zeros(dims[i+1]))
            self.W.append(W)
            self.b.append(b)
        if activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'relu':
            self.act = F.relu
        else:
            raise ValueError('unknown activation')
        self.to(device)

    def forward(self, x):
        r = x.view(x.shape[0], -1)
        for i in range(self.n_layers):
            r = self.act(F.linear(r, self.W[i], self.b[i]))
        return r

    def infer_hidden_states(self, x_batch, y_batch, n_steps=20, lr_infer=0.1, lambda_sup=1.0):
        B = x_batch.shape[0]
        r0 = x_batch.view(B, -1).to(self.device)
        with torch.no_grad():
            r_init = []
            prev = r0
            for i in range(self.n_layers):
                out = self.act(F.linear(prev, self.W[i], self.b[i]))
                r_init.append(out.detach().clone())
                prev = out.detach().clone()
        r_vars = [nn.Parameter(r.clone().detach()) for r in r_init]
        for p in r_vars:
            p.requires_grad = True
        inner_opt = torch.optim.SGD(r_vars, lr=lr_infer)
        W_detached = [w.detach() for w in self.W]
        b_detached = [b.detach() for b in self.b]
        ce = nn.CrossEntropyLoss()
        for step in range(n_steps):
            inner_opt.zero_grad()
            loss_pred = 0.0
            prev = r0
            for i in range(self.n_layers):
                pred = self.act(F.linear(prev, W_detached[i], b_detached[i]))
                err = (r_vars[i] - pred)
                loss_pred = loss_pred + torch.mean(err.pow(2))
                prev = r_vars[i]
            loss_sup = ce(r_vars[-1], y_batch.to(self.device))
            total_loss = loss_pred + lambda_sup * loss_sup
            total_loss.backward()
            inner_opt.step()
        with torch.no_grad():
            inferred = [p.detach().clone() for p in r_vars]
        return inferred

    def outer_update_weights(self, x_batch, inferred_states, optimizer, lambda_sup=1.0):
        B = x_batch.shape[0]
        r0 = x_batch.view(B, -1).to(self.device)
        targets = [r.detach() for r in inferred_states]
        prev = r0
        loss_pred = 0.0
        for i in range(self.n_layers):
            pred = self.act(F.linear(prev, self.W[i], self.b[i]))
            err = (targets[i] - pred)
            loss_pred = loss_pred + torch.mean(err.pow(2))
            prev = targets[i]
        total_loss = loss_pred
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return float(total_loss.item())

# %%
if __name__ == "__main__":
    # Save module file (optional)
    import inspect
    module_code = inspect.getsource(PCN_MLP)
    with open('model_pcn.py', 'w') as f:
        f.write('import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n')
        f.write(module_code)
    print("Wrote model_pcn.py")
