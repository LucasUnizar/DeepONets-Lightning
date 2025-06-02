import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import wandb


class DeepONet(pl.LightningModule):
    def __init__(self, m=100, P=100, hidden_dim=128, trunk_layers=3, branch_layers=3):
        """
        Args:
            m: Number of input function sensors
            P: Number of evaluation points
            hidden_dim: Dimension of hidden layers
            trunk_layers: Number of layers in trunk network (minimum 1)
            branch_layers: Number of layers in branch network (minimum 1)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Validate layer counts
        if trunk_layers < 1 or branch_layers < 1:
            raise ValueError("Both trunk_layers and branch_layers must be at least 1")

        # Branch net (processes input function)
        branch_modules = []
        in_features = m
        for i in range(branch_layers):
            branch_modules.append(nn.Linear(in_features, hidden_dim))
            branch_modules.append(nn.Tanh())
            in_features = hidden_dim
        
        # Remove last Tanh if more than 1 layer (optional choice)
        if branch_layers > 1:
            branch_modules = branch_modules[:-1]
            
        self.branch = nn.Sequential(*branch_modules)

        # Trunk net (processes coordinates)
        trunk_modules = []
        in_features = 2  # For (x,t) coordinates
        for i in range(trunk_layers):
            trunk_modules.append(nn.Linear(in_features, hidden_dim))
            trunk_modules.append(nn.Tanh())
            in_features = hidden_dim
        
        # Remove last Tanh if more than 1 layer (optional choice)
        if trunk_layers > 1:
            trunk_modules = trunk_modules[:-1]
            
        self.trunk = nn.Sequential(*trunk_modules)

        # Initialize weights
        self._init_weights()

        # Store validation examples for visualization
        self.validation_examples = []

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_func, coords):
        # Process input function through branch net
        branch_out = self.branch(input_func)  # [batch_size, hidden_dim]
        
        # Process coordinates through trunk net
        trunk_out = self.trunk(coords)  # [batch_size, num_points, hidden_dim] or [batch_size, hidden_dim]
        
        # Ensure shapes match for dot product
        if len(trunk_out.shape) == 2:
            trunk_out = trunk_out.unsqueeze(1)  # Add point dimension if needed
            
        # Dot product and sum
        output = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)  # [batch_size, num_points]
        
        return output.squeeze(-1)  # Ensure output is [batch_size] or [batch_size, num_points]

    def training_step(self, batch, batch_idx):
        input_func = batch['input_func']
        coords = batch['coords']
        solution = batch['solution']

        pred = self(input_func, coords)
        
        # Ensure shapes match
        pred = pred.view(-1, 1)  # [batch_size, 1]
        solution = solution.view(-1, 1)  # [batch_size, 1]
        
        loss = F.mse_loss(pred, solution)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_func = batch['input_func']
        coords = batch['coords']
        solution = batch['solution']

        pred = self(input_func, coords)
        
        # Ensure shapes match
        pred = pred.view(-1, 1)
        solution = solution.view(-1, 1)
        
        loss = F.mse_loss(pred, solution)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_func = batch['input_func']
        coords = batch['coords']
        solution = batch['solution']

        pred = self(input_func, coords)
        
        # Ensure shapes match
        pred = pred.view(-1, 1)
        solution = solution.view(-1, 1)
        
        loss = F.mse_loss(pred, solution)

        # Calculate relative L2 error
        relative_l2 = torch.norm(pred - solution, p=2) / torch.norm(solution, p=2)
        self.log('test_loss', loss)
        self.log('test_relative_l2', relative_l2)

        # Plot and save example predictions for the first test batch
        if batch_idx == 0:
            self._plot_test_example(coords.cpu(), solution.cpu(), pred.cpu())
            
        return loss

    def _plot_test_example(self, coords, solution, pred):
        """Plot and save test examples showing true solution vs prediction"""
        # Convert to numpy arrays
        coords = coords.numpy()
        solution = solution.numpy().flatten()
        pred = pred.numpy().flatten()
        
        # Create grid for surface plot
        grid_x, grid_t = np.mgrid[coords[:,0].min():coords[:,0].max():100j, 
                          coords[:,1].min():coords[:,1].max():100j]
        
        # Interpolate true solution
        grid_solution = griddata(coords, solution, (grid_x, grid_t), method='cubic')
        
        # Interpolate prediction
        grid_pred = griddata(coords, pred, (grid_x, grid_t), method='cubic')
        
        # Create figure with 4 subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Scatter plot of true solution
        ax1 = fig.add_subplot(221)
        sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=solution, cmap='plasma', s=10)
        plt.colorbar(sc1, ax=ax1)
        ax1.set_title('True Solution (Scatter)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        
        # Scatter plot of prediction
        ax2 = fig.add_subplot(222)
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=pred, cmap='plasma', s=10)
        plt.colorbar(sc2, ax=ax2)
        ax2.set_title('Predicted Solution (Scatter)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        
        # Surface plot of true solution
        ax3 = fig.add_subplot(223, projection='3d')
        surf1 = ax3.plot_surface(grid_x, grid_t, grid_solution, cmap='plasma',
                                linewidth=0, antialiased=False, alpha=0.8)
        fig.colorbar(surf1, ax=ax3, shrink=0.5, aspect=5)
        ax3.set_title('True Solution (Surface)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('u(x,t)')
        
        # Surface plot of prediction
        ax4 = fig.add_subplot(224, projection='3d')
        surf2 = ax4.plot_surface(grid_x, grid_t, grid_pred, cmap='plasma',
                                linewidth=0, antialiased=False, alpha=0.8)
        fig.colorbar(surf2, ax=ax4, shrink=0.5, aspect=5)
        ax4.set_title('Predicted Solution (Surface)')
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        ax4.set_zlabel('u(x,t)')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('test_prediction_example.png')
        plt.close(fig)
        
        # Optionally log to wandb if available
        if wandb.run is not None:
            wandb.log({"test_prediction_example": wandb.Image(fig)})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }