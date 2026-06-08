import os
import psutil
import gc
import torch
import sys
import queue
import threading
import numpy as np

sys.path.append('C:\\Users\\reyse\\OneDrive\\Desktop\\Bayesian-AI')

from training.rl_engine.network_research_A import ResearchANetwork
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem
from training.rl_engine.train_gpu_research_A import run_quadrant_sim

device = torch.device('cuda')
master_net = ResearchANetwork(channels=8).to(device)
optimizer = torch.optim.Adam(master_net.parameters(), lr=0.01)

days = ['2025_01_02', '2025_01_03']
fps = MultiDayForwardPassSystem(
    atlas_root='DATA/ATLAS',
    features_root='DATA/ATLAS/FEATURES_5s_v2',
    labels_csv='DATA/ATLAS/regime_labels_2d.csv',
    days=days
)

process = psutil.Process(os.getpid())

print("Initial RAM:", process.memory_info().rss / 1024 / 1024, "MB")

for epoch in range(10):
    run_quadrant_sim(fps, master_net, optimizer, vtrace=None, config={}, device=device, epoch_idx=epoch, N_AGENTS=128, is_eval=False)
    gc.collect()
    print(f"Epoch {epoch} RAM:", process.memory_info().rss / 1024 / 1024, "MB")
