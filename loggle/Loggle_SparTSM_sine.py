import numpy as np
import torch
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from matplotlib.lines import Line2D


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def g(t):
    return -(t-1)*t

def dg(t):
    return -2*t+1

def f(x):
    n, d, nt = x.shape
    pairwise_features = torch.zeros((n, d * (d + 1) // 2, nt))

    idx = 0
    for i in range(d):
        for j in range(i, d):
            pairwise_features[:, idx, :] = x[:, i, :] * x[:, j, :]
            idx += 1

    return pairwise_features

def phi(fXqt, t, sigma = .1):
    Kt = torch.exp(-(t[:, None] - t[:, None].T)**2/sigma**2/2)
    Eqtfx = fXqt[0, :, :] @ Kt / torch.sum(Kt, 1)
    return fXqt - Eqtfx

def fourierfea(b, t):
    freq = torch.arange(b, dtype=torch.float32)
    fourier_features = torch.cat([freq * torch.cos(freq * t), -freq * torch.sin(freq * t)], dim=-1)
    return fourier_features

def dfourierfea(b, t):
    freq = torch.arange(b, dtype=torch.float32)
    fourier_features = torch.cat([-freq**2 *torch.sin(freq * t), -freq**2 *torch.cos(freq * t)], dim=-1)
    return fourier_features

def fea(t, b, d):
    F = torch.zeros(t.shape[0], 2*b, d)
    for i in range(d):
        F[:, :, i] = fourierfea(b, t)
    return F

def dfea(t, b, d):
    F = torch.zeros(t.shape[0], 2*b, d)
    for i in range(d):
        F[:, :, i] = dfourierfea(b, t)
    return F

def obj(alpha, phiXqt, fXqt, F, dF):
    n = phiXqt.shape[0]
    dthetat = torch.sum(alpha * F, 1).transpose(1, 0)
    d2thetat = torch.sum(alpha * dF, 1).transpose(1, 0)

    obj1 = torch.mean((torch.sum(phiXqt * dthetat, 1)**2) @ g(t), 0)
    obj2 = torch.ones(1,n) @ torch.sum(fXqt * dthetat, 1) / n @ dg(t)
    obj3 = torch.ones(1,n) @ torch.sum(fXqt * d2thetat, 1) / n @ g(t)

    return obj1 + 2*obj2 + 2*obj3

def train(X, t, lmbd = .001, num_epochs=100, learning_rate=0.0001):
    phiXqt = phi(f(X), t)
    fXqt = f(X)
    n = phiXqt.shape[0]
    d = phiXqt.shape[1]

    k = 5  # number of fourier features
    F = fea(t[:, None], k, d)
    dF = dfea(t[:, None], k, d)

    alpha = torch.zeros(2*k, f(X).shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pen = sum(torch.norm(alpha[:, i], 2) for i in range(d))
        loss = obj(alpha, phiXqt, fXqt, F, dF) + lmbd * pen
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return alpha

torch.manual_seed(2)
n = 1
nt = 5000
d = 20
G = torch.rand(d, d) + torch.eye(d) < .02
G = G + G.t()

Xqt = torch.zeros(n, d, nt)
t = torch.rand(nt)
for i in range(nt):
    nz = torch.zeros(1, d)
    mu = torch.zeros(1, d) + t[i]*nz
    Theta = torch.eye(d)
    base = torch.randn(d, 2*d)
    base = base @ base.T/2/d
    Theta = Theta + base
    Theta.fill_diagonal_(2.0)
    Theta[G == 1] = .5 * torch.sin(t[i]*10)
    Cov = torch.inverse(Theta)
    Xqt[:, :, i] = torch.distributions.MultivariateNormal(mu, Cov).sample((n,)).squeeze()

alpha_hat = train(Xqt, t, lmbd=1700, learning_rate=0.001, num_epochs=1000)

t_plot = torch.linspace(0, 1, 1000)
F = fea(t_plot[:, None], 5, f(Xqt).shape[1])
dthetat = torch.sum(alpha_hat * F, 1).transpose(1, 0)

import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
# -- Create subplots --
plt.rcParams.update({
    'font.size': 20,          # Base font size
    'axes.titlesize': 20,     # Title font size
    'axes.labelsize': 18,     # Axes labels font size
    'xtick.labelsize': 18,    # X tick labels font size
    'ytick.labelsize': 18,    # Y tick labels font size
    'legend.fontsize': 20,    # Legend font size
})


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))




legend_label_in_support_tsm = (r'$\partial_t \hat{\boldsymbol{\Theta}}_{ij}(t), \; '
                               r'ij \in \mathrm{supp}(\partial_t \boldsymbol{\Theta}^*(t))$')
legend_label_not_in_support_tsm = (r'$\partial_t \hat{\boldsymbol{\Theta}}_{ij}(t), \; '
                                   r'ij \notin \mathrm{supp}(\partial_t \boldsymbol{\Theta}^*(t))$')
idx = 0
for i in range(d):
    for j in range(i, d):
        if i != j and G[i, j] != 1:
            ax1.plot(t_plot.detach().numpy(), dthetat[idx, :].detach().numpy(), c='b', alpha=0.5, linewidth=2)
        idx += 1

idx = 0
for i in range(d):
    for j in range(i, d):
        if i != j and G[i, j] == 1:
            ax1.plot(t_plot.detach().numpy(), dthetat[idx, :].detach().numpy(), c='r', alpha=0.5, linewidth=4)
        idx += 1

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.8, 0.8)
ax1.set_xlabel('Time')
ax1.set_ylabel(r'$\partial_t \hat{\boldsymbol{\Theta}}_{ij}(t)$')
ax1.set_title(r'\textbf{SparTSM}')
legend_elements = [Line2D([0], [0], color='r', lw=2, label=legend_label_in_support_tsm),
                   Line2D([0], [0], color='b', lw=2, label=legend_label_not_in_support_tsm)]

ax1.legend(handles=legend_elements, loc='upper right', fontsize=20)



# Set random seeds for reproducibility
# torch.manual_seed(2)
# np.random.seed(2)

# Generate the graph G
G = (torch.rand(d, d) + torch.eye(d)) < 0.02
G = G | G.t() 
G_np = G.numpy()


edge_indices = np.argwhere(np.triu(np.ones_like(G_np), k=1)) 
num_edges = edge_indices.shape[0]

# Generate data
Xqt = torch.zeros(n, d, nt)
t = torch.rand(nt)
base = torch.randn(d, 2 * d)
base = base @ base.T / (2 * d)

for i in range(nt):
    nz = torch.zeros(1, d)
    mu = torch.zeros(1, d) + t[i] * nz
    Theta = torch.eye(d)
    Theta = Theta + base  
    Theta.fill_diagonal_(2.0)
    Theta[G] = 0.5 * torch.sin(t[i] * 10)  
    Cov = torch.inverse(Theta)
    Xqt[:, :, i] = torch.distributions.MultivariateNormal(mu.squeeze(), Cov).sample((n,))


X = Xqt.permute(2, 0, 1).reshape(nt * n, d).numpy().T  
def run_loggle_in_r(X, h=0.1, d=0.01, lambda_val=0.03):
    numpy2ri.activate()
    importr('loggle')
    with localconverter(default_converter + numpy2ri.converter):
        X_r = robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    r_code = f"""
    pos <- 1:{X.shape[1]}
    result <- loggle(X, pos, h = {h}, d = {d}, lambda = {lambda_val},
                     fit.type = "pseudo", refit = TRUE,
                     num.thread = parallel::detectCores() - 1)
    omega_list <- result$Omega
    """
    robjects.globalenv['X'] = X_r
    try:
        robjects.r(r_code)
    except Exception as e:
        print(f"Error in executing R code: {e}")
        return None
    omega_list_r = robjects.r['omega_list']
    omega_list_py = []
    for omega_r in omega_list_r:
        with localconverter(default_converter + numpy2ri.converter):
            omega_dense = np.array(robjects.r['as.matrix'](omega_r))
            omega_list_py.append(omega_dense)
    omega_array = np.array(omega_list_py)
    return omega_array

omega_estimations = run_loggle_in_r(X)

legend_label_in_support_loggle = (r'$\hat{\boldsymbol{\Theta}}_{ij}(t), \; '
                                  r'ij \in \mathrm{supp}(\partial_t \boldsymbol{\Theta}^*(t))$')
legend_label_not_in_support_loggle = (r'$\hat{\boldsymbol{\Theta}}_{ij}(t), \; '
                                      r'ij \notin \mathrm{supp}(\partial_t \boldsymbol{\Theta}^*(t))$')

def calculate_and_plot_change_point(omega_estimations, edge_indices, G_np, ax):
    omega_list_py = omega_estimations
    N = len(omega_list_py)
    num_edges = edge_indices.shape[0]

    omega_sequence_matrix = np.array([
        [omega_list_py[t][i, j] for t in range(N)]
        for i, j in edge_indices
    ]) 

    changing_edges = np.array([G_np[i, j] for i, j in edge_indices], dtype=bool)

    for idx in range(num_edges):
        if not changing_edges[idx]:  # Plot non-changing edges first
            omega_ij = omega_sequence_matrix[idx]
            ax.plot(np.linspace(0, 1, N), omega_ij, color='blue', linestyle='-', alpha=0.5, linewidth=2)

    for idx in range(num_edges):
        if changing_edges[idx]:  # Plot changing edges second
            omega_ij = omega_sequence_matrix[idx]
            ax.plot(np.linspace(0, 1, N), omega_ij, color='red', linestyle='-', alpha=0.5, linewidth=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\hat{\boldsymbol{\Theta}}_{ij}(t)$')
    ax.set_title(r'\textbf{Loggle}')

    legend_elements = [Line2D([0], [0], color='r', lw=2, label=legend_label_in_support_loggle),
                   Line2D([0], [0], color='b', lw=2, label=legend_label_not_in_support_loggle)]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

if omega_estimations is not None:
    calculate_and_plot_change_point(omega_estimations, edge_indices, G_np, ax2)

plt.tight_layout()
plt.show()
