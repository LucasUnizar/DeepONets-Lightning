from dataset import ReactionDiffusionDataModule
from model import DeepONet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib
import wandb

matplotlib.use('Agg')  # Use non-interactive backend for plotting

if __name__ == '__main__':
    # Initialize data module
    input_sensors = 100 # f(x)
    output_sensors = 100 # u(x,t)
    project='DeepONet-Reaction-Diffusion'
    mat_file_path = 'reaction_diffusion_dataset.mat'  # Replace with your actual .mat file path
    batch_size=10000
    data_module = ReactionDiffusionDataModule(mat_file_path, batch_size=batch_size, sampled=output_sensors)
    data_module.setup('fit')  # This initializes the datasets

    # Get the length of the training dataset
    train_dataset_length = len(data_module.train_dataset)
    wandb_logger = pl.loggers.WandbLogger(project=project, log_model=True)

    # Initialize model
    model = DeepONet(m=input_sensors, P=output_sensors, hidden_dim=50, trunk_layers=5, branch_layers=5)
    iterations = 120000  # Your desired number of iterations
    steps_per_epoch = train_dataset_length // batch_size
    # max_epochs = iterations // steps_per_epoch + 1
    max_epochs = 2  # Set a maximum number of epochs for training
    print(f'Max epoch as: {max_epochs}')

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='deeponet-{epoch:02d}-{val_loss:.2e}',
        save_top_k=1,
        mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module, ckpt_path='best')