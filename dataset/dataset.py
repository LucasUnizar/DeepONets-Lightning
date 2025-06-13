import torch
from torch.utils.data import Dataset
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

class ReactionDiffusionDataset(Dataset):
    def __init__(self, mat_file_path, split='train', sampled=100):
        """
        Args:
            mat_file_path: Path to the .mat file
            split: 'train', 'val', or 'test'
            sampled: Number of points to sample per trajectory
        """
        print(f"\nLoading data from {mat_file_path} for {split} split")
        data = sio.loadmat(mat_file_path)

        # Convert numpy arrays to torch tensors
        u_train = torch.from_numpy(data['u_train']).float()  # Input functions [Traj*Sampled, m]
        y_train = torch.from_numpy(data['y_train']).float()  # Coordinates [Traj*Sampled, 2]
        s_train = torch.from_numpy(data['s_train']).float()  # Solutions [Traj*Sampled, 1]
        
        # Reshape from [Traj*Sampled, ...] to [Traj, Sampled, ...]
        num_trajectories = u_train.shape[0] // sampled
        u_train = u_train.view(num_trajectories, sampled, -1)  # [Traj, Sampled, m]
        y_train = y_train.view(num_trajectories, sampled, -1)  # [Traj, Sampled, 2]
        s_train = s_train.view(num_trajectories, sampled, -1)  # [Traj, Sampled, 1]
        
        # Split trajectories (80-10-10 split)
        num_trajectories = u_train.shape[0]
        train_end = int(0.8 * num_trajectories)
        val_end = train_end + int(0.1 * num_trajectories)

        if split == 'train':
            u_train = u_train[:train_end]
            y_train = y_train[:train_end]
            s_train = s_train[:train_end]
        elif split == 'val':
            u_train = u_train[train_end:val_end]
            y_train = y_train[train_end:val_end]
            s_train = s_train[train_end:val_end]
        else:  # test
            u_train = u_train[val_end:]
            y_train = y_train[val_end:]
            s_train = s_train[val_end:]
        
        # Flatten back to [Batch, ...] format (Batch = Traj * Sampled)
        self.u = u_train.view(-1, u_train.shape[-1])  # [Traj*Sampled, m]
        self.y = y_train.view(-1, y_train.shape[-1])  # [Traj*Sampled, 2]
        self.s = s_train.view(-1, s_train.shape[-1])  # [Traj*Sampled, 1]

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            'input_func': self.u[idx],  # [m]
            'coords': self.y[idx],      # [2]
            'solution': self.s[idx]    # [1]
        }

# Data Module
class DataModule(pl.LightningDataModule):
    def __init__(self, mat_file_path, batch_size=32, sampled=100):
        super().__init__()
        self.mat_file_path = mat_file_path
        self.sampled = sampled
        self.batch_size = batch_size
        print(f"\nInitializing DataModule with batch_size: {batch_size}")

    def setup(self, stage=None):
        print(f"\nSetting up DataModule for stage: {stage}")
        if stage == 'fit' or stage is None:
            self.train_dataset = ReactionDiffusionDataset(
                self.mat_file_path, 'train', self.sampled)
            self.val_dataset = ReactionDiffusionDataset(
                self.mat_file_path, 'val', self.sampled)
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")

        if stage == 'test' or stage is None:
            self.test_dataset = ReactionDiffusionDataset(
                self.mat_file_path, 'test', self.sampled)
            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        print("Creating train dataloader")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=2)

    def val_dataloader(self):
        print("Creating val dataloader")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=2)

    def test_dataloader(self):
        print("Creating test dataloader")
        return DataLoader(self.test_dataset, batch_size=self.sampled, 
                         num_workers=2)