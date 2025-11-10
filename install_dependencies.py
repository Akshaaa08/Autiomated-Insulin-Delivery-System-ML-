"""
Installation script for TFT Glucose Prediction
Run this FIRST before running the main script
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    print(f"\nInstalling {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print(f"✓ {package} installed")

def uninstall_package(package):
    """Uninstall a package using pip"""
    print(f"\nUninstalling {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package, "-y", "-q"])
        print(f"✓ {package} uninstalled")
    except:
        print(f"  {package} was not installed")

print("="*80)
print("GLUCOSE PREDICTION - DEPENDENCY INSTALLATION")
print("="*80)

# Step 1: Uninstall conflicting versions
print("\n[Step 1/3] Removing conflicting packages...")
uninstall_package("lightning")
uninstall_package("pytorch-lightning")
uninstall_package("pytorch-forecasting")

# Step 2: Install PyTorch (CPU version - change if you have CUDA)
print("\n[Step 2/3] Installing PyTorch...")
print("Note: This installs CPU version. For GPU, modify the command.")
try:
    install_package("torch")
    install_package("torchvision")
except:
    print("Installing PyTorch from index...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

# Step 3: Install compatible versions
print("\n[Step 3/3] Installing PyTorch Lightning and PyTorch Forecasting...")
install_package("pytorch-lightning==2.0.0")
install_package("pytorch-forecasting")

# Step 4: Install other dependencies
print("\nInstalling other dependencies...")
packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "scikit-learn",
]

for package in packages:
    install_package(package)

# Verify installation
print("\n" + "="*80)
print("VERIFYING INSTALLATION")
print("="*80)

try:
    import torch
    import pytorch_lightning as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ PyTorch Lightning: {pl.__version__}")
    print(f"✓ Pandas: {pd.__version__}")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ Matplotlib: (installed)")
    print(f"✓ Scikit-learn: (installed)")
    print(f"✓ PyTorch Forecasting: (installed)")
    
    print("\n" + "="*80)
    print("✓ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
    print("="*80)
    print("\nYou can now run the main glucose prediction script.")
    
except Exception as e:
    print(f"\n✗ Error during verification: {e}")
    print("\nTry manually installing:")
    print("  pip install pytorch-lightning==2.0.0")
    print("  pip install pytorch-forecasting")
    print("  pip install pandas numpy matplotlib scikit-learn")