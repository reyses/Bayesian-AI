import os
import sys
import time
import torch
import numpy as np
from sklearn.linear_model import ElasticNetCV
from group_lasso import GroupLasso

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.math.fista_gpu import elasticnet_fista_cv, group_lasso_fista_cv

def test_elasticnet_parity():
    print("--- Testing ElasticNet Parity ---")
    N, p = 150, 50
    np.random.seed(42)
    X = np.random.randn(N, p)
    # create sparse true w
    true_w = np.zeros(p)
    true_w[np.random.choice(p, 10, replace=False)] = np.random.randn(10) * 5
    y = X @ true_w + np.random.randn(N) * 0.1
    
    print("Running Sklearn CPU...")
    t0 = time.time()
    enet = ElasticNetCV(cv=3, l1_ratio=0.5, fit_intercept=False)
    enet.fit(X, y)
    t1 = time.time()
    print(f"CPU Time: {t1 - t0:.4f}s")
    
    active_cpu = np.where(np.abs(enet.coef_) > 1e-5)[0]
    
    print("Running FISTA GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    
    t0 = time.time()
    w_gpu, best_alpha = elasticnet_fista_cv(X_t, y_t, l1_ratio=0.5, cv=3)
    t1 = time.time()
    print(f"GPU CV Time: {t1 - t0:.4f}s")
    
    from core_v2.math.fista_gpu import elasticnet_fista
    w_gpu_direct = elasticnet_fista(X_t, y_t, enet.alpha_, l1_ratio=0.5)
    
    print(f"CPU best_alpha: {enet.alpha_:.6f}")
    print(f"GPU best_alpha: {best_alpha:.6f}")
    print(f"CPU alpha_max: {np.max(enet.alphas_):.6f}, min: {np.min(enet.alphas_):.6f}")
    
    w_cpu_from_gpu = w_gpu_direct.cpu().numpy().flatten()
    active_gpu = np.where(np.abs(w_cpu_from_gpu) > 1e-5)[0]
    
    print(f"CPU Active Set: {active_cpu}")
    print(f"GPU Active Set: {active_gpu}")
    
    overlap = len(np.intersect1d(active_cpu, active_gpu))
    union = len(np.union1d(active_cpu, active_gpu))
    print(f"Jaccard Similarity: {overlap / union * 100:.2f}%")
    
def test_group_lasso_parity():
    print("\n--- Testing Group Lasso Parity ---")
    N, p = 150, 60
    np.random.seed(42)
    X = np.random.randn(N, p)
    groups = np.repeat(np.arange(p // 3), 3)[:p]
    
    true_w = np.zeros(p)
    true_w[:6] = np.random.randn(6) * 5
    y = X @ true_w + np.random.randn(N) * 0.1
    
    print("Running Sklearn CPU...")
    t0 = time.time()
    gl = GroupLasso(groups=groups, group_reg=0.01, l1_reg=0.0, n_iter=100, fit_intercept=False, supress_warning=True)
    gl.fit(X, y)
    t1 = time.time()
    print(f"CPU Time: {t1 - t0:.4f}s")
    
    active_cpu = np.where(np.abs(gl.coef_.flatten()) > 1e-5)[0]
    
    print("Running FISTA GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    
    t0 = time.time()
    w_gpu, best_alpha = group_lasso_fista_cv(X_t, y_t, groups, cv=3)
    t1 = time.time()
    print(f"GPU Time: {t1 - t0:.4f}s")
    
    w_cpu_from_gpu = w_gpu.cpu().numpy()
    active_gpu = np.where(np.abs(w_cpu_from_gpu) > 1e-5)[0]
    
    print(f"CPU Active Set: {active_cpu}")
    print(f"GPU Active Set: {active_gpu}")

if __name__ == "__main__":
    test_elasticnet_parity()
    test_group_lasso_parity()
