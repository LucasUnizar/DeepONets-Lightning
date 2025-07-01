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
    def __init__(self, args, m=100, P=100, hidden_dim=128, trunk_layers=3, branch_layers=3):
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
        # External arguments
        self.args = args
        self.max_epochs = args.max_epochs
        self.domain = args.domain if hasattr(args, 'domain') else [0, 1]
        self.time_domain = args.time_domain if hasattr(args, 'time_domain') else [0, 1]
        
        # Determine activation function
        if hasattr(args, 'activation_function') and args.activation_function == 'cosine_tanh':
            self.activation = lambda x: F.tanh(F.cos(x)) # Apply cos then tanh
            print("Using Cosine-Tanh activation function.")
        else:
            self.activation = nn.Tanh() # Default to Tanh
            print("Using Tanh activation function.")
        
        # Validate layer counts
        if trunk_layers < 1 or branch_layers < 1:
            raise ValueError("Both trunk_layers and branch_layers must be at least 1")

        # Branch net (processes input function)
        branch_modules = []
        in_features = m
        for i in range(branch_layers):
            branch_modules.append(nn.Linear(in_features, hidden_dim))
            branch_modules.append(self.activation) # Use the selected activation
            in_features = hidden_dim
        
        # Remove last activation if more than 1 layer (optional choice, often done for last layer output)
        if branch_layers > 1:
            branch_modules = branch_modules[:-1]
            
        self.branch = nn.Sequential(*branch_modules)

        # Trunk net (processes coordinates)
        trunk_modules = []
        in_features = 2  # For (x,t) coordinates
        for i in range(trunk_layers):
            trunk_modules.append(nn.Linear(in_features, hidden_dim))
            trunk_modules.append(self.activation) # Use the selected activation
            in_features = hidden_dim
        
        # Remove last activation if more than 1 layer (optional choice)
        if trunk_layers > 1:
            trunk_modules = trunk_modules[:-1]
            
        self.trunk = nn.Sequential(*trunk_modules)

        self.bias = nn.Parameter(torch.zeros(1))  # Initialize bias as a learnable parameter

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
        
        # Add bias term
        output = output + self.bias  # [batch_size, num_points]
    
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
        
        # Calculate relative L2 error
        relative_l2 = torch.norm(pred - solution, p=2) / (torch.norm(solution, p=2) + 1e-8)  # Added small epsilon to avoid division by zero
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_relative_l2', relative_l2, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_func = batch['input_func']
            coords = batch['coords']
            solution = batch['solution']

            pred = self(input_func, coords)
            
            # Ensure shapes match
            pred = pred.view(-1, 1)
            solution = solution.view(-1, 1)
            
            loss = F.mse_loss(pred, solution)
            
            # Calculate relative L2 error
            relative_l2 = torch.norm(pred - solution, p=2) / (torch.norm(solution, p=2) + 1e-8)
            
            # Log metrics
            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
            self.log('val_relative_l2', relative_l2, on_epoch=True)
            
            return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
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
                self._plot_test_example(
                    coords.cpu(), 
                    solution.cpu(), 
                    pred.cpu(),
                    input_func[0,:].cpu()  # Pass the input function to the plotting function
                )

            return loss

    def on_test_end(self, num_points=100, idx_arg=0):
        with torch.no_grad():
            print("Generating extra dense test example for visualization...")
            for idx, batch in enumerate(self.trainer.datamodule.test_dataloader()):
                input_func = batch['input_func']
                f_x = input_func[0, :].unsqueeze(0)  # [1, m]
                domain = self.domain
                time = self.time_domain

                x = torch.linspace(domain[0], domain[1], num_points).to(self.device)
                t = torch.linspace(time[0], time[1], num_points).to(self.device)

                # Create 2D grids of x and t
                X_grid, T_grid = torch.meshgrid(x, t, indexing='ij') # 'ij' for matrix-style indexing

                # Flatten the grids and combine them
                coords = torch.stack([X_grid.flatten(), T_grid.flatten()], dim=1).squeeze().to(self.device)  # [num_points^2, 2]
                input_func_dense = f_x.repeat(coords.shape[0], 1).to(self.device)  # [num_points^2, m]

                pred = self(input_func_dense, coords)
                if idx == idx_arg:
                    self._plot_test_example_solution_only(coords.cpu(), pred.cpu(), input_func_dense[0,:].cpu())
                    print("Dense test example plotted and saved.")
                    return
                    

    def _plot_test_example(self, coords, solution, pred, ic):
        """Plot and save test examples showing true solution vs prediction with colored IC line"""
        # Convert to numpy arrays
        coords = coords.numpy()
        solution = solution.numpy().flatten()
        pred = pred.numpy().flatten()
        ic = ic.numpy().flatten()

        # Create figure with adjusted aspect ratio for horizontal layout
        fig = plt.figure(figsize=(24, 8))  # Wider figure for horizontal subplots

        # Create gridspec for 1 row, 3 columns
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

        # Get common normalization for all plots using the solution
        norm = plt.Normalize(vmin=min(solution.min(), pred.min()), 
                            vmax=max(solution.max(), pred.max()))

        # Scatter plot of true solution with colored IC line at t=0
        ax1 = fig.add_subplot(gs[0])
        sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=solution, cmap='plasma', norm=norm, s=10)
        
        # Add colored IC line at t=0 (1D plot with same colormap)
        x_coords = np.linspace(self.domain[0], self.domain[1], ic.shape[0]) 
        for i in range(len(x_coords)-1):
            ax1.plot(x_coords[i:i+2], [0, 0], 
                    color=plt.cm.plasma(norm(ic[i])), 
                    linewidth=2)
        
        plt.colorbar(sc1, ax=ax1)
        ax1.set_title('True Solution with IC at t=0')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_aspect('auto')

        # Scatter plot of prediction with colored IC line at t=0
        ax2 = fig.add_subplot(gs[1])
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=pred, cmap='plasma', norm=norm, s=10)
        
        # Add colored IC line at t=0 (1D plot with same colormap)
        for i in range(len(x_coords)-1):
            ax2.plot(x_coords[i:i+2], [0, 0], 
                    color=plt.cm.plasma(norm(ic[i])), 
                    linewidth=2)
        
        plt.colorbar(sc2, ax=ax2)
        ax2.set_title('Predicted Solution with IC at t=0')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_aspect('auto')

        # Absolute error plot
        ax3 = fig.add_subplot(gs[2])
        error = np.abs(pred - solution)
        sc3 = ax3.scatter(coords[:, 0], coords[:, 1], c=error, cmap='Reds', s=10)
        plt.colorbar(sc3, ax=ax3)
        ax3.set_title('Absolute Error (Prediction vs Ground Truth)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_aspect('auto')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('outputs/test_prediction_example.png')
        plt.close(fig)
        
        # Optionally log to wandb if available
        if wandb.run is not None:
            wandb.log({"test_prediction_example": wandb.Image(fig)})
    

    def _plot_test_example_solution_only(self, coords, solution, ic):
        """Plot and save test examples showing true solution, its surface map, and IC."""
        # Convert to numpy arrays
        coords = coords.numpy()
        solution = solution.numpy().flatten()
        ic = ic.numpy().flatten()

        # Create grid for surface plot
        grid_x, grid_t = np.mgrid[coords[:,0].min():coords[:,0].max():100j,
                        coords[:,1].min():coords[:,1].max():100j]

        # Interpolate true solution
        grid_solution = griddata(coords, solution, (grid_x, grid_t), method='cubic')

        # Create figure with 2 subplots
        fig = plt.figure(figsize=(12, 8)) # Adjusted figure size for 2 plots

        # Create common normalization
        norm = plt.Normalize(vmin=min(solution.min(), ic.min()),
                            vmax=max(solution.max(), ic.max()))

        # Scatter plot of true solution with IC
        ax1 = fig.add_subplot(121) # 1 row, 2 columns, 1st plot
        sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=solution, cmap='plasma', norm=norm, s=10)
        
        # Add colored IC line at t=0
        x_coords = np.linspace(coords[:,0].min(), coords[:,0].max(), len(ic))
        for i in range(len(x_coords)-1):
            ax1.plot(x_coords[i:i+2], [0, 0], 
                    color=plt.cm.plasma(norm(ic[i])), 
                    linewidth=2)
        
        plt.colorbar(sc1, ax=ax1)
        ax1.set_title('Solution with IC at t=0 (Scatter)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')

        # Surface plot of true solution with IC
        ax2 = fig.add_subplot(122, projection='3d') # 1 row, 2 columns, 2nd plot
        surf1 = ax2.plot_surface(grid_x, grid_t, grid_solution, cmap='plasma', norm=norm,
                                linewidth=0, antialiased=False, alpha=0.8)
        
        # Add colored IC line at t=0 on surface plot
        t_min = coords[:,1].min()
        ax2.plot(x_coords, np.zeros_like(x_coords), ic,
                color='black', linewidth=1.5, alpha=0.7)
        
        # Add colored dots for IC values
        sc_ic = ax2.scatter(x_coords, np.zeros_like(x_coords), ic,
                          c=ic, cmap='plasma', norm=norm, s=20)
        
        fig.colorbar(surf1, ax=ax2, shrink=0.5, aspect=5)
        ax2.set_title('Solution with IC at t=0 (Surface)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('u(x,t)')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('outputs/true_solution_example.png') # Changed filename
        plt.close(fig)
        
        # Optionally log to wandb if available
        if wandb.run is not None:
            wandb.log({"true_solution_example": wandb.Image(fig)})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          # or 'max' depending on your metric
            factor=0.8,          # Reduce LR by this factor
            patience=50,          # Number of epochs with no improvement after which LR will be reduced
            min_lr=1e-8         # Minimum learning rate
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_relative_l2_epoch',  # Change this to your metric, e.g., 'val_accuracy' if mode='max'
                'interval': 'epoch',
                'frequency': 1
            }
        }