import torch
import numpy as np

def soft_threshold_gpu(x, lam):
    """
    Applies soft-thresholding element-wise on GPU tensor.
    S(x, lam) = sign(x) * max(|x| - lam, 0)
    """
    return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - lam)

def block_soft_threshold_gpu(x, group_indices, lam):
    """
    Applies block soft-thresholding for Group Lasso.
    x: shape (p,) or (batch, p)
    group_indices: list of tensors, where each tensor contains indices for a group.
    """
    out = x.clone()
    for idx in group_indices:
        if out.dim() == 1:
            block = out[idx]
            norm = torch.norm(block)
            scale = torch.nn.functional.relu(1.0 - lam / (norm + 1e-8))
            out[idx] = block * scale
        else:
            # Batched block thresholding (x is shape (p, alphas))
            block = out[idx, :]
            norm = torch.norm(block, dim=0, keepdim=True)
            scale = torch.nn.functional.relu(1.0 - lam / (norm + 1e-8))
            out[idx, :] = block * scale
    return out

def group_lasso_fista(X, y, group_indices, lam, max_iter=500, tol=1e-4):
    """
    FISTA implementation for Group Lasso on GPU.
    X: (N, p)
    y: (N, 1)
    lam: regularization penalty
    """
    device = X.device
    N, p = X.shape
    if y.dim() == 1:
        y = y.view(-1, 1)
    
    # Calculate Lipschitz constant (max eigenvalue of X^T X)
    XtX = X.T @ X
    L = torch.linalg.norm(XtX, ord=2).item() / N
    step_size = 0.5 / (L + 1e-6) # 0.5 guarantees Nesterov convergence
    
    w = torch.zeros((p, 1), device=device)
    z = torch.zeros((p, 1), device=device)
    t = 1.0
    
    for i in range(max_iter):
        w_old = w.clone()
        
        # Gradient of 1/2N ||Xz - y||^2
        grad = (X.T @ (X @ z - y)) / N
        
        # Proximal gradient step
        w_unreg = z - step_size * grad
        w = block_soft_threshold_gpu(w_unreg, group_indices, step_size * lam)
        
        # Nesterov momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = w + ((t - 1.0) / t_new) * (w - w_old)
        t = t_new
        
        if torch.max(torch.abs(w - w_old)) < tol:
            break
            
    return w

def elasticnet_fista(X, y, lam, l1_ratio=0.5, max_iter=500, tol=1e-4):
    """
    FISTA implementation for ElasticNet on GPU.
    """
    device = X.device
    N, p = X.shape
    if y.dim() == 1:
        y = y.view(-1, 1)
    
    # L2 Ridge expansion (ElasticNet equivalent formula)
    lam_l2 = lam * (1.0 - l1_ratio)
    lam_l1 = lam * l1_ratio
    
    XtX = (X.T @ X) / N
    XtX.diagonal().add_(lam_l2) # Add L2 penalty to diagonal
    Xty = (X.T @ y) / N
    
    L = torch.linalg.norm(XtX, ord=2).item()
    step_size = 0.5 / (L + 1e-6)
    
    w = torch.zeros((p, 1), device=device)
    z = torch.zeros((p, 1), device=device)
    t = 1.0
    
    for i in range(max_iter):
        w_old = w.clone()
        
        grad = XtX @ z - Xty
        
        w_unreg = z - step_size * grad
        w = soft_threshold_gpu(w_unreg, step_size * lam_l1)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = w + ((t - 1.0) / t_new) * (w - w_old)
        t = t_new
        
        if torch.max(torch.abs(w - w_old)) < tol:
            break
            
    return w

def get_group_indices(groups):
    """Helper to convert group assignments into index tensors for the GPU."""
    unique_groups = np.unique(groups)
    group_indices = []
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        group_indices.append(torch.tensor(idx, dtype=torch.long))
    return group_indices

def get_kfold_splits(N, cv=3):
    """Generates train/val indices for K-Fold CV without leakage."""
    indices = np.arange(N)
    fold_sizes = np.full(cv, N // cv, dtype=int)
    fold_sizes[:N % cv] += 1
    current = 0
    splits = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        splits.append((train_idx, val_idx))
        current = stop
    return splits

def elasticnet_fista_cv(X, y, l1_ratio=0.5, cv=3, alphas=100, eps=1e-3):
    """
    K-Fold Cross Validated ElasticNet on GPU.
    Computes all alphas simultaneously via Tensor Core batching.
    """
    N, p = X.shape
    device = X.device
    if y.dim() == 1:
        y = y.view(-1, 1)
    
    Xty = torch.abs(X.T @ y) / N
    alpha_max = torch.max(Xty).item() / max(l1_ratio, 1e-3)
    alpha_min = alpha_max * eps
    alpha_path = torch.logspace(np.log10(alpha_max), np.log10(alpha_min), steps=alphas, device=device)
    
    splits = get_kfold_splits(N, cv)
    mse_path = torch.zeros((alphas, cv), device=device)
    
    for i, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        XtX_tr = (X_tr.T @ X_tr) / len(train_idx)
        Xty_tr = (X_tr.T @ y_tr) / len(train_idx) # (p, 1)
        L = torch.linalg.norm(XtX_tr, ord=2).item()

        lam_l2 = alpha_path * (1.0 - l1_ratio) # (alphas,)
        lam_l1 = (alpha_path * l1_ratio).unsqueeze(0) # (1, alphas)
        # The smooth part is (1/2N)||Xz-y||^2 + (lam_l2/2)||z||^2, whose gradient
        # has Lipschitz constant L + lam_l2 (lam_l2 enters the gradient via the
        # z*lam_l2 term below, NOT the Gram diagonal). A scalar 1/L step ignored
        # lam_l2, so high-alpha columns (large lam_l2) overshot the stability
        # bound and diverged to NaN -> argmin(mean_mse) then returned the NaN
        # column (alpha_max) and the final fit was all-zeros. Per-column step
        # 1/(L + lam_l2) respects the true Lipschitz for every alpha at once.
        step_size = (1.0 / (L + lam_l2 + 1e-6)).unsqueeze(0) # (1, alphas)

        w = torch.zeros((p, alphas), device=device)
        z = torch.zeros((p, alphas), device=device)
        t = 1.0
        
        for _ in range(500):
            w_old = w.clone()
            
            # Batched gradient: XtX @ z is (p, alphas)
            # z * lam_l2 adds the L2 penalty to the diagonal mathematically
            grad = (XtX_tr @ z) + (z * lam_l2) - Xty_tr
            
            w_unreg = z - step_size * grad
            w = soft_threshold_gpu(w_unreg, step_size * lam_l1)
            
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            z = w + ((t - 1.0) / t_new) * (w - w_old)
            t = t_new
            
            if torch.max(torch.abs(w - w_old)) < 1e-4: break
            
        preds = X_val @ w # (N_val, alphas)
        mse = torch.mean((y_val - preds)**2, dim=0) # (alphas,)
        mse_path[:, i] = mse
            
    mean_mse = torch.mean(mse_path, dim=1)
    # Defense in depth: a diverged/unstable column must never win the argmin.
    mean_mse = torch.nan_to_num(mean_mse, nan=float('inf'), posinf=float('inf'))
    best_alpha_idx = torch.argmin(mean_mse)
    best_alpha = alpha_path[best_alpha_idx].item()

    # Fit final model
    w_final = elasticnet_fista(X, y, best_alpha, l1_ratio, max_iter=1000)
    return w_final.squeeze(), best_alpha

def group_lasso_fista_cv(X, y, groups, cv=3, alphas=20, eps=1e-2):
    """
    K-Fold Cross Validated Group Lasso on GPU.
    Computes all alphas simultaneously.
    """
    N, p = X.shape
    device = X.device
    if y.dim() == 1:
        y = y.view(-1, 1)
    group_indices = get_group_indices(groups)
    
    alpha_max = torch.max(torch.abs(X.T @ y)).item() / N
    alpha_min = alpha_max * eps
    alpha_path = torch.logspace(np.log10(alpha_max), np.log10(alpha_min), steps=alphas, device=device)
    
    splits = get_kfold_splits(N, cv)
    mse_path = torch.zeros((alphas, cv), device=device)
    device_groups = [g.to(device) for g in group_indices]
    
    for i, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        XtX_tr = X_tr.T @ X_tr
        L = torch.linalg.norm(XtX_tr, ord=2).item() / len(train_idx)
        step_size = 1.0 / (L + 1e-6)
        
        w = torch.zeros((p, alphas), device=device)
        z = torch.zeros((p, alphas), device=device)
        t = 1.0
        lam = alpha_path.unsqueeze(0) # (1, alphas)
        
        for _ in range(500):
            w_old = w.clone()
            
            err = (X_tr @ z) - y_tr # (N, alphas)
            grad = (X_tr.T @ err) / len(train_idx) # (p, alphas)
            
            w_unreg = z - step_size * grad
            w = block_soft_threshold_gpu(w_unreg, device_groups, step_size * lam)
            
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            z = w + ((t - 1.0) / t_new) * (w - w_old)
            t = t_new
            
            if torch.max(torch.abs(w - w_old)) < 1e-4: break
            
        preds = X_val @ w
        mse = torch.mean((y_val - preds)**2, dim=0)
        mse_path[:, i] = mse
            
    mean_mse = torch.mean(mse_path, dim=1)
    best_alpha_idx = torch.argmin(mean_mse)
    best_alpha = alpha_path[best_alpha_idx].item()
    
    w_final = group_lasso_fista(X, y, device_groups, best_alpha, max_iter=1000)
    return w_final.squeeze(), best_alpha
