#!/usr/bin/env python3
"""
DeepONet Training Script for Reaction-Diffusion Problems

This script provides a configurable training pipeline for DeepONet models
with Weights & Biases integration and checkpointing.
"""

import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, lr_monitor
from pytorch_lightning.loggers import WandbLogger
import matplotlib
import os
import torch

from src.dataset.dataset import DataModule
from src.model.DeepONet import DeepONet

# Set non-interactive backend for plotting
matplotlib.use('Agg')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DeepONet on Reaction-Diffusion data')
    
    # Mode control
    parser.add_argument('--train', action='store_true',
                       help='Train the model (default)')
    parser.add_argument('--test', action='store_true',
                       help='Test the model with saved weights')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pre-trained model weights for transfer learning')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=r'data/reaction_diffusion_dataset_N5000_P100_L0.20_100x100.mat',
                       help='Path to the .mat data file')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for training')
    parser.add_argument('--input_sensors', type=int, default=100,
                       help='Number of input sensors (m)')
    parser.add_argument('--output_sensors', type=int, default=100,
                       help='Number of output sensors (P)')
    parser.add_argument('--domain', type=list, default=[0, 1],
                       help='Domain of the input data as a list [min, max]')
    parser.add_argument('--time_domain', type=list, default=[0, 1],
                       help='Time domain of the input data as a list [min, max]')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=50,
                       help='Hidden dimension size')
    parser.add_argument('--trunk_layers', type=int, default=5,
                       help='Number of trunk network layers')
    parser.add_argument('--branch_layers', type=int, default=5,
                       help='Number of branch network layers')
    
    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=1,
                       help='Maximum number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for the optimizer')
    parser.add_argument('--iterations', type=int, default=120000,
                       help='Desired number of iterations (alternative to max_epochs)')
    parser.add_argument('--val_check_interval', type=int, default=1,
                       help='How often to check validation (in epochs)')
    
    # Logging/checkpoint arguments
    parser.add_argument('--project', type=str, default='DeepONets-Jaca',
                       help='W&B project name')
    parser.add_argument('--name', type=str, default='DeepONet-test',
                       help='W&B run name')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--weights_path', type=str, default=None,
                       help='Path to saved weights for testing')
    parser.add_argument('--log_model', action='store_true',
                       help='Log model to W&B')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                       help='Patience for early stopping (None to disable)')
    
    # Hardware arguments
    parser.add_argument('--accelerator', type=str, default='auto',
                       help='Type of accelerator to use (auto, cpu, gpu, etc.)')
    parser.add_argument('--devices', type=str, default='auto',
                       help='Devices to use for training')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate mode arguments
    if args.train and args.test:
        raise ValueError("Cannot specify both --train and --test flags")
    if not args.train and not args.test:
        args.train = True  # Default to training mode
    
    # Setup data module
    data_module = DataModule(
        mat_file_path=args.data_path,
        batch_size=args.batch_size,
        sampled=args.output_sensors
    )
    data_module.setup('fit' if args.train else 'test')

    #data_module.plot_solution_with_ic(trajectory_idx=0, input_domain=args.domain)
    
    # Initialize model
    model = DeepONet(
        args=args,
        m=args.input_sensors,
        P=args.output_sensors,
        hidden_dim=args.hidden_dim,
        trunk_layers=args.trunk_layers,
        branch_layers=args.branch_layers
    )
    
    if args.test:
        # Testing mode - load saved weights
        if args.pretrained is not None:
            weights_path = args.pretrained
        else:
            weights_path = args.weights_path or Path(args.save_dir) / 'last.ckpt'
            if not Path(weights_path).exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}. "
                                    "Please specify with --weights_path or train a model first.")
        
        print(f"Loading model weights from: {weights_path}")
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        trainer.test(model, datamodule=data_module, ckpt_path=weights_path)
        return
    
    # Training mode
    print(f"Training configuration:")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Input sensors: {args.input_sensors}")
    print(f"- Output sensors: {args.output_sensors}")
    print(f"- Training samples: {len(data_module.train_dataset)}")
    
    # Calculate steps per epoch and adjust max_epochs if needed
    train_dataset_length = len(data_module.train_dataset)
    steps_per_epoch = train_dataset_length // args.batch_size
    
    # If max_epochs is not specified, calculate from iterations
    if args.max_epochs is None and args.iterations is not None:
        args.max_epochs = args.iterations // steps_per_epoch + 1
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Max epochs: {args.max_epochs}")
    
    # Setup logging and callbacks
    wandb_logger = WandbLogger(
        project=args.project,
        log_model=args.log_model,
        name=args.name,
        config=vars(args)  # Log all arguments
    )
    
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,
        filename=f'{args.name}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    lr_monitor_callback = lr_monitor.LearningRateMonitor(logging_interval='epoch')  
    callbacks = [checkpoint_callback, lr_monitor_callback]
    
    if args.early_stop_patience is not None:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.early_stop_patience,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.devices,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=args.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=torch.backends.cudnn.deterministic,
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Test the model after training
    trainer.test(model, datamodule=data_module, ckpt_path='best')


if __name__ == '__main__':
    main()