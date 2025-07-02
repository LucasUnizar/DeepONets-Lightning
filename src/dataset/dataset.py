import torch
from torch.utils.data import Dataset
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

class LoadDataset(Dataset):
    def __init__(self, mat_file_path, split='train', sampled=100):
        """
        Args:
            mat_file_path: Path to the .mat file
            split: 'train', 'val', or 'test'
            sampled: Number of points to sample per trajectory
        """
        print(f"\nLoading data from {mat_file_path} for {split} split")
        data = sio.loadmat(mat_file_path)
        
        # Print all available keys in the .mat file
        print(f"Available keys in .mat file: {list(data.keys())}")

        # Convert numpy arrays to torch tensors
        u_train = torch.from_numpy(data['u_train']).float()  # Input functions [Traj*Sampled, m]
        y_train = torch.from_numpy(data['y_train']).float()  # Coordinates [Traj*Sampled, 2]
        s_train = torch.from_numpy(data['s_train']).float()  # Solutions [Traj*Sampled, 1]
        
        if split == 'train':
            print(f"\nOriginal shapes before reshaping:")
            print(f"u_train shape: {u_train.shape} (input functions)")
            print(f"y_train shape: {y_train.shape} (coordinates)")
            print(f"s_train shape: {s_train.shape} (solutions)")
        
        # Reshape from [Traj*Sampled, ...] to [Traj, Sampled, ...]
        num_trajectories = u_train.shape[0] // sampled
        u_train = u_train.view(num_trajectories, sampled, -1)  # [Traj, Sampled, m]
        y_train = y_train.view(num_trajectories, sampled, -1)  # [Traj, Sampled, 2]
        s_train = s_train.view(num_trajectories, sampled, -1)  # [Traj, Sampled, 1]
        
        if split == 'train':
            print(f"\nAfter reshaping to [Trajectories, Sampled, Features]:")
            print(f"u_train shape: {u_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"s_train shape: {s_train.shape}")
        
        # Split trajectories (80-10-10 split)
        num_trajectories = u_train.shape[0]
        train_end = int(0.8 * num_trajectories)
        val_end = train_end + int(0.1 * num_trajectories)

        if split == 'train':
            u_train = u_train[:train_end]
            y_train = y_train[:train_end]
            s_train = s_train[:train_end]
            print(f"\nAfter selecting {split} split ({train_end} trajectories):")
        elif split == 'val':
            u_train = u_train[train_end:val_end]
            y_train = y_train[train_end:val_end]
            s_train = s_train[train_end:val_end]
            print(f"\nAfter selecting {split} split ({val_end - train_end} trajectories):")
        else:  # test
            u_train = u_train[val_end:]
            y_train = y_train[val_end:]
            s_train = s_train[val_end:]
            print(f"\nAfter selecting {split} split ({num_trajectories - val_end} trajectories):")
            
        print(f"u_train shape: {u_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"s_train shape: {s_train.shape}")

        # Flatten back to [Batch, ...] format (Batch = Traj * Sampled)
        self.u = u_train.view(-1, u_train.shape[-1])  # [Traj*Sampled, m]
        self.y = y_train.view(-1, y_train.shape[-1])  # [Traj*Sampled, 2]
        self.s = s_train.view(-1, s_train.shape[-1])  # [Traj*Sampled, 1]
        
        print(f"\nFinal flattened shapes for {split} dataset:")
        print(f"Input functions (u) shape: {self.u.shape}")
        print(f"Coordinates (y) shape: {self.y.shape}")
        print(f"Solutions (s) shape: {self.s.shape}")

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            'input_func': self.u[idx],  # [m]
            'coords': self.y[idx],      # [2]
            'solution': self.s[idx]     # [1]
        }

# Data Module
class DataModule(pl.LightningDataModule):
    def __init__(self, mat_file_path, batch_size=32, sampled=100):
        super().__init__()
        self.mat_file_path = mat_file_path
        self.sampled = sampled
        self.batch_size = batch_size
        print(f"\nInitializing DataModule with batch_size: {batch_size}, sampled: {sampled}")

    def setup(self, stage=None):
        print(f"\nSetting up DataModule for stage: {stage}")
        if stage == 'fit' or stage is None:
            self.train_dataset = LoadDataset(
                self.mat_file_path, 'train', self.sampled)
            self.val_dataset = LoadDataset(
                self.mat_file_path, 'val', self.sampled)
            print(f"\nDataset sizes:")
            print(f"Train dataset size: {len(self.train_dataset)} samples")
            print(f"Val dataset size: {len(self.val_dataset)} samples")

        if stage == 'test' or stage is None:
            self.test_dataset = LoadDataset(
                self.mat_file_path, 'test', self.sampled)
            print(f"Test dataset size: {len(self.test_dataset)} samples")

    def plot_solution_with_ic(self, trajectory_idx=0, input_domain=[0,1]):
        """
        Plot solution with initial condition in 2D scatter view.
        
        Args:
            split: 'val'
            trajectory_idx: Index of the trajectory to plot
            input_domain: Domain [min, max] for the x-dimension
        """
        # Get the appropriate dataset
        batch = self.val_dataset[trajectory_idx]
        
        # Select the specific trajectory (reshape if needed)
        coords = batch['coords'] # [1, 2]
        solution = batch['solution'] # u(x,t)
        ic = batch['input_func'] # Initial condition at t=0
        
        # Convert to numpy arrays
        coords = coords.numpy()
        solution = solution.numpy().flatten()
        ic = ic.numpy().flatten()

        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create normalization
        norm = plt.Normalize(vmin=min(solution.min(), ic.min()),
                            vmax=max(solution.max(), ic.max()))

        # Scatter plot of solution
        sc = plt.scatter(coords[0], coords[1], c=solution, 
                        cmap='plasma', norm=norm, s=10)
        
        # Add colored IC line at t=0 using specified input domain
        x_coords = np.linspace(input_domain[0], input_domain[1], len(ic))
        for i in range(len(x_coords)-1):
            plt.plot(x_coords[i:i+2], [0, 0], 
                    color=plt.cm.plasma(norm(ic[i])), 
                    linewidth=3)
        
        plt.colorbar(sc, label='u(x,t)')
        plt.title(f'Solution with IC (Trajectory {trajectory_idx}')
        plt.xlabel('x (domain: [{}, {}])'.format(input_domain[0], input_domain[1]))
        plt.ylabel('t')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def train_dataloader(self):
        print("\nCreating train dataloader")
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=0)
        print(f"Train batches: {len(loader)} (batch_size: {self.batch_size})")
        return loader

    def val_dataloader(self):
        print("\nCreating val dataloader")
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=0)
        print(f"Val batches: {len(loader)} (batch_size: {self.batch_size})")
        return loader

    def test_dataloader(self):
        print("\nCreating test dataloader")
        loader = DataLoader(self.test_dataset, batch_size=self.sampled, 
                         num_workers=0)
        print(f"Test batches: {len(loader)} (batch_size: {self.sampled})")
        return loader