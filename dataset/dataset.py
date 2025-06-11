import torch
from torch.utils.data import Dataset
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class ReactionDiffusionDataset(Dataset):
    def __init__(self, mat_file_path, split='train'):
        """
        Args:
            mat_file_path: Path to the .mat file
            split: 'train', 'val', or 'test'
        """
        print(f"\nLoading data from {mat_file_path} for {split} split")
        data = sio.loadmat(mat_file_path)

        # Convert numpy arrays to torch tensors
        u_train = torch.from_numpy(data['u_train']).float()  # Input functions [N, m]
        y_train = torch.from_numpy(data['y_train']).float()  # Coordinates [N, P, 2]
        s_train = torch.from_numpy(data['s_train']).float()  # Solutions [N, P]

        # Split data (80-10-10 split)
        num_samples = u_train.shape[0]
        train_end = int(0.8 * num_samples)
        val_end = train_end + int(0.1 * num_samples)


        if split == 'train':
            self.u = u_train[:train_end]
            self.y = y_train[:train_end]
            self.s = s_train[:train_end]
        elif split == 'val':
            self.u = u_train[train_end:val_end]
            self.y = y_train[train_end:val_end]
            self.s = s_train[train_end:val_end]
        else:  # test
            self.u = u_train[val_end:]
            self.y = y_train[val_end:]
            self.s = s_train[val_end:]

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            'input_func': self.u[idx],  # [m]
            'coords': self.y[idx],      # [P, 2]
            'solution': self.s[idx]     # [P]
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
            self.train_dataset = ReactionDiffusionDataset(self.mat_file_path, 'train')
            self.val_dataset = ReactionDiffusionDataset(self.mat_file_path, 'val')
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")

        if stage == 'test' or stage is None:
            self.test_dataset = ReactionDiffusionDataset(self.mat_file_path, 'test')
            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        print("Creating train dataloader")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        print("Creating val dataloader")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        print("Creating test dataloader")
        return DataLoader(self.test_dataset, batch_size=self.sampled, num_workers=2)