from setuptools import setup, find_packages

setup(
    name='DeepONetsLightning',
    version='0.1.0',
    author='Lucas T',
    author_email='ltesan@unizar.es',
    description='DeepONets implementation using PyTorch Lightning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LucasUnizar/DeepONets-Lightning',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'pytorch-lightning',
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'wandb'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
