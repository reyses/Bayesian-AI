import h5py
import numpy as np
import os

class HDF5ShadowQueue:
    """
    Zero-Copy Out-Of-Core Experience Replay Buffer.
    Streams completed trajectories to an HDF5 database on the C: drive (M.2 SSD)
    to bypass the 12GB PyTorch VRAM limit.
    V2 Native: Stores the 8x23 V2 grids and the L0 features separately.
    """
    
    def __init__(self, db_path="C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/training/rl_engine/experiences.h5"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Creates the extensible HDF5 dataset structure."""
        if not os.path.exists(self.db_path):
            with h5py.File(self.db_path, 'w') as f:
                # Store V2 Grids: [Batch, TFs(8), SeqLen(60), Features(23)]
                f.create_dataset('v2_grids', shape=(0, 8, 60, 23), maxshape=(None, 8, 60, 23), dtype='float32', chunks=True)
                
                # Store L0 Features: [Batch, SeqLen(60), 3]
                f.create_dataset('l0_features', shape=(0, 60, 3), maxshape=(None, 60, 3), dtype='float32', chunks=True)
                
                # Store actions: [Batch, 1]
                f.create_dataset('actions', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
                
                # Store regret/rewards: [Batch, 1]
                f.create_dataset('regrets', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
                
                # Store off-policy probabilities for V-trace
                f.create_dataset('behavior_pi', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)

    def write_terminal_trajectory(self, v2_grid, l0_feature, action, regret, behavior_pi):
        """
        Appends a fully resolved trajectory (where Structural Exit was reached)
        directly to the HDF5 disk storage.
        """
        with h5py.File(self.db_path, 'a') as f:
            d_grids = f['v2_grids']
            d_l0 = f['l0_features']
            d_actions = f['actions']
            d_regrets = f['regrets']
            d_pi = f['behavior_pi']

            idx = d_grids.shape[0]
            
            # Resize datasets
            d_grids.resize(idx + 1, axis=0)
            d_l0.resize(idx + 1, axis=0)
            d_actions.resize(idx + 1, axis=0)
            d_regrets.resize(idx + 1, axis=0)
            d_pi.resize(idx + 1, axis=0)
            
            # Append data directly to disk
            d_grids[idx] = v2_grid
            d_l0[idx] = l0_feature
            d_actions[idx] = action
            d_regrets[idx] = regret
            d_pi[idx] = behavior_pi

    def get_dataset_size(self):
        if not os.path.exists(self.db_path):
            return 0
        with h5py.File(self.db_path, 'r') as f:
            return f['v2_grids'].shape[0]
