# %%
import torch 
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    n, d = x.shape
    pairwise_features = torch.zeros((n, d * (d + 1) // 2))
    
    idx = 0
    for i in range(d):
        for j in range(i, d):
            pairwise_features[:, idx] = x[:, i] * x[:, j]
            idx += 1
            
    return pairwise_features

# %%
def train(Xp, Xq, lmbd = .001):
    fXp = f(Xp).detach().numpy()
    fXq = f(Xq).detach().numpy()
    delta = torch.zeros(fXp.shape[1], 1).detach().numpy()

    def obj(delta, fXp, fXq):
        return - np.mean(fXp @ delta) + np.log(np.mean(np.exp(fXq @ delta)))

    def grad(delta, fXp, fXq):
        exp_fXq_delta = np.exp(fXq @ delta)
        return - np.mean(fXp, axis=0) + np.sum(fXq * exp_fXq_delta, axis=0) / np.sum(exp_fXq_delta)
    
    def proximal_operator_l1(v, alpha):
        return np.sign(v) * np.maximum(np.abs(v) - alpha, 0)

    def proximal_gradient_descent(lambd, x0, lr, max_iter):
        x = x0.copy()
        for i in range(max_iter):
            gradient = grad(x, fXp, fXq)[: , None]
            x = x - lr * gradient
            x = proximal_operator_l1(x, lr * lambd)
            
            # print(f"Iteration {i}: Loss = {obj(x, fXp, fXq)}")
        return x
    
    sol = proximal_gradient_descent(lmbd, delta, 0.01, 10000)
    return torch.tensor(sol)

    
# %%
torch.manual_seed(2)
n = 500
nt = 1000
d = 40

G = torch.rand(d, d) + torch.eye(d) < .023
G = G + G.t()
plt.imshow(G)

Xqt = torch.zeros(n, d, nt)
t = torch.rand(nt)

print("generating samples for each time point ..., may take a while...", end="")
base = torch.randn(d, 2*d)
base = base @ base.T/2/d
for i in range(nt):
    nz = torch.zeros(1, d)
    mu = torch.zeros(1, d) + t[i]*nz
    Theta = torch.eye(d)
    Theta = Theta + base
    Theta.fill_diagonal_(2.0)
    Theta[G == 1] = .45*t[i]
                
    Cov = torch.inverse(Theta)
    Xqt[:, :, i] = torch.distributions.MultivariateNormal(mu, Cov).sample((n,)).squeeze()

print("done.")
# %%
print('training model...')

ROC_points = []

# get 10 values of lambda from 1e-3 to 1e-1
for lmbd in np.logspace(-3, 0, 20):

    Xp = Xqt[:, :, 0]
    Xq = Xqt[:, :, -1]
    alpha_hat = train(Xp, Xq, lmbd= lmbd)

    # plt.plot(alpha_hat.detach().numpy())

    #reconstruct alpha to Theta 
    Theta = torch.zeros(d, d)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            Theta[i, j] = alpha_hat[idx]
            Theta[j, i] = alpha_hat[idx]
            idx += 1

    print(Theta)

    FP = 0
    TP = 0

    for i in range(d):
        for j in range(i+1, d):
            if torch.abs(Theta[i, j]) > 1e-6:
                if G[i, j] == 1:
                    TP += 1
                else:
                    FP += 1
            
        
    NP = torch.sum(G)/2
    NN = d*(d-1)/2 - NP

    FPR = FP / NN   
    TPR = TP / NP

    print([FPR, TPR])
    ROC_points.append(torch.tensor([FPR, TPR]))

ROC_points = torch.stack(ROC_points)

# %%
# plot ROC curve

plt.plot(ROC_points[:, 0], ROC_points[:, 1], marker='o')

# %%

torch.save(ROC_points, 'ratio_ROC_points.pt')
# %%
