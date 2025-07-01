"""
This code implements a ResNet-inspired neural network for regression tasks, 
focusing on predicting aerosol mixing state metrics (Chi) based on chemical 
and environmental features. The dataset includes PartMC simulations for 
training and validation, along with MEGAPOLI data for testing. Hyperparameter 
optimization is performed using Optuna, ensuring an efficient and robust model 
configuration. The process includes data preprocessing, model definition, 
training, validation, and evaluation.

Author: Fei Jiang, The University of Manchester
"""


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import random


# Set random seed for reproducibility
def set_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)


# Define features used for model input
features = ['Temperature','RH','Bulk_DP','Num_Conc','OA', 'SO4', 'Cl', 'NH4', 'NO3']

# Load datasets
partmc_train_data = pd.read_csv('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/partmc_train.csv')
partmc_test_data = pd.read_csv('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/partmc_test.csv')
partmc_valid_data = pd.read_csv('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/partmc_valid.csv')


# Prepare input features (X) and target variable (y) for training, validation, and testing
var = 'CCN_VS_1.0'
X_train = partmc_train_data[features]
y_train = partmc_train_data[var]

X_valid = partmc_valid_data[features]
y_valid = partmc_valid_data[var]

X_test = partmc_test_data[features]
y_test = partmc_test_data[var]


# Standardize the input features to follow a standard normal distribution
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_valid = scaler_X.transform(X_valid)
X_test = scaler_X.transform(X_test)

# Convert datasets to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


# Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensors to the specified device
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_valid_tensor = X_valid_tensor.to(device)
y_valid_tensor = y_valid_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


# Create PyTorch dataset for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)


# Define the ResNet block
class ResNetBlock(nn.Module):
    """
    A single ResNet block with two fully connected layers and a residual connection.
    """
    def __init__(self, hidden_size):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  # Add residual connection
        out = self.relu(out)
        return out
    

# Define the ResNet-like model
class ResNet(nn.Module):
    """
    A ResNet-inspired model for regression tasks.
    """
    def __init__(self, in_features, num_blocks, hidden_size):
        super(ResNet, self).__init__()
        self.fc_in = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.fc_out = nn.Linear(hidden_size, 1)  # Output a single value for regression
    
    def forward(self, x):
        out = self.fc_in(x)
        out = self.relu(out)
        out = self.blocks(out)
        out = self.fc_out(out)
        return out
    

# Define the Optuna objective function for hyperparameter optimization
def objective(trial):
    """
    Objective function for Optuna to optimize the hyperparameters of the ResNet model.

    Args:
        trial (optuna.trial.Trial): A single optimization trial.

    Returns:
        float: The validation loss.
    """
    # Define hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512, 1024])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    num_blocks = trial.suggest_int('num_blocks', 10, 20)

    # Create DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = ResNet(X_train.shape[1], num_blocks, hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 40
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_valid_tensor).view(-1)
            val_loss = criterion(val_outputs, y_valid_tensor).item()

        # Track the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

# Perform hyperparameter optimization using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Output the best hyperparameters
print('Best hyperparameters:', study.best_params)

