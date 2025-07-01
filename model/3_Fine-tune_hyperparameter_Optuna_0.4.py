import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import random
import optuna


# Function to set random seed for reproducibility
def set_seed(seed):
    """
    Sets the random seed for reproducibility across numpy, random, and PyTorch.

    Parameters:
    - seed (int): The seed value to ensure reproducibility.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)  # CPU
    if torch.cuda.is_available():  # GPU
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the random seed to ensure reproducibility
seed = 42  # Seed value for reproducibility
set_seed(seed)

features = ['Temperature', 'RH', 'Bulk_DP', 'Num_Conc', 'OA', 'SO4', 'Cl', 'NH4', 'NO3']


# Load the dataset used for training the original model for Normalizing
partmc_train_data = pd.read_csv('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/partmc_train.csv')
X_train = partmc_train_data[features]


# Load the fine-tuning datasets
arm_train_data = pd.read_csv('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/setpoint_0.4_train.csv') # e.g. Use 50% fine-tuning training dataset to fine-tune foundation model
arm_test_data = pd.read_csv('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/setpoint_0.4_test.csv')

def load_data(arm_train_data, arm_test_data):

    # Prepare arm data
    X_arm_train = arm_train_data[features]
    y_arm_train = arm_train_data['N_CCN']
    X_arm_test = arm_test_data[features]
    y_arm_test = arm_test_data['N_CCN']

    # Standardize the data using the scaler from the original model's training data
    scaler_X = StandardScaler()
    X_train2 = scaler_X.fit_transform(X_train)  #  Fit on the original training data, X_train2 ensures no need to reload the original dataset (PartMC)
    X_arm_train = scaler_X.transform(X_arm_train)
    X_arm_test = scaler_X.transform(X_arm_test)
    return X_arm_train, y_arm_train, X_arm_test, y_arm_test


# Define the ResNet-like model architecture
class ResNetBlock(nn.Module):
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
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_features, num_blocks, hidden_size):
        super(ResNet, self).__init__()
        self.fc_in = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.fc_in(x)
        out = self.relu(out)
        out = self.blocks(out)
        out = self.fc_out(out)
        return out

# Define the Optuna objective function for hyperparameter optimization
def objective(trial):
    X_arm_train, y_arm_train, X_arm_test, y_arm_test = load_data(arm_train_data, arm_test_data)
    
    X_arm_train_tensor = torch.tensor(X_arm_train, dtype=torch.float32)
    y_arm_train_tensor = torch.tensor(y_arm_train, dtype=torch.float32)
    X_arm_test_tensor = torch.tensor(X_arm_test, dtype=torch.float32)
    y_arm_test_tensor = torch.tensor(y_arm_test, dtype=torch.float32)

    arm_train_dataset = TensorDataset(X_arm_train_tensor, y_arm_train_tensor)
    train_loader = DataLoader(arm_train_dataset, batch_size=8, shuffle=True)
    
    input_size = X_arm_train_tensor.shape[1]
    num_blocks = 13
    hidden_size = 1024
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)   
    num_frozen_blocks = trial.suggest_int("num_frozen_blocks", 9, 12)  # Optimize number of frozen layers
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)  # Optimize L2 regularization
    
    model = ResNet(input_size, num_blocks, hidden_size)
    model.load_state_dict(torch.load('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/model/Foundation_Model_0.4.pth'))
    
    # Freeze the selected number of layers
    for i, block in enumerate(model.blocks):
        if i < num_frozen_blocks:
            for param in block.parameters():
                param.requires_grad = False
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    num_epochs = 30
    best_mse = float('inf')
    no_improve = 0
    patience = 3
    best_weights = None

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_arm_test_tensor).view(-1).numpy()
            y_true = y_arm_test_tensor.numpy()
            test_mse = mean_squared_error(y_true, predictions)

        if test_mse < best_mse:
            best_mse = test_mse
            best_weights = model.state_dict().copy()
            no_improve = 0
        else: 
            no_improve += 1
            if no_improve >= patience:
                break
    
    model.load_state_dict(best_weights)

    return best_mse

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=40)
    print("Best hyperparameters:", study.best_params)