import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def fit_logistic_regression(X_tensor, y_tensor, max_iter=100):
    model = LogisticRegressionModel(X_tensor.shape[1]).to(X_tensor.device)
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = nn.BCELoss()
    
    def closure():
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        final_loss = criterion(outputs, y_tensor).item()
        
    return model, final_loss

def pytorch_stepwise_forward(X, y, n_features_to_select=15, device='cuda'):
    """
    Performs Stepwise Forward Selection using PyTorch on GPU.
    X: numpy array (N, F)
    y: numpy array (N,)
    """
    if not torch.cuda.is_available():
        device = 'cpu'
        
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    
    N, F = X_tensor.shape
    selected_features = []
    remaining_features = list(range(F))
    
    # Standardize X for stability
    X_mean = X_tensor.mean(dim=0, keepdim=True)
    X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
    X_tensor = (X_tensor - X_mean) / X_std
    
    # Calculate Null Model Loss (only bias)
    null_model = LogisticRegressionModel(1).to(device)
    null_optimizer = optim.LBFGS(null_model.parameters(), lr=1.0, max_iter=100)
    null_criterion = nn.BCELoss()
    dummy_X = torch.zeros((N, 1)).to(device)
    
    def null_closure():
        null_optimizer.zero_grad()
        loss = null_criterion(null_model(dummy_X), y_tensor)
        loss.backward()
        return loss
        
    null_optimizer.step(null_closure)
    with torch.no_grad():
        ll_null = null_criterion(null_model(dummy_X), y_tensor).item()
    
    best_loss = ll_null
    
    for step in range(n_features_to_select):
        best_feature_this_step = -1
        best_loss_this_step = float('inf')
        
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_candidate = X_tensor[:, candidate_features]
            
            _, loss = fit_logistic_regression(X_candidate, y_tensor, max_iter=20)
            
            if loss < best_loss_this_step:
                best_loss_this_step = loss
                best_feature_this_step = feature
                
        if best_feature_this_step != -1:
            selected_features.append(best_feature_this_step)
            remaining_features.remove(best_feature_this_step)
            best_loss = best_loss_this_step
        else:
            break
            
    pseudo_r2 = 1 - (best_loss / ll_null) if ll_null != 0 else 0
    return selected_features, pseudo_r2
