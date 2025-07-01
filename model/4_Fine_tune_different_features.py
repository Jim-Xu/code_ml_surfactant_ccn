import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

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

def load_partial_state_dict(model, checkpoint_path):
    """迁移除输入层外的参数，适应不同输入特征数量。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = model.state_dict()
    # 拷贝blocks和fc_out参数，忽略fc_in
    for k in checkpoint:
        if k.startswith('blocks') or k.startswith('fc_out'):
            if k in new_state_dict and checkpoint[k].shape == new_state_dict[k].shape:
                new_state_dict[k] = checkpoint[k]
    model.load_state_dict(new_state_dict)
    return model


set_points = [0.1, 0.2, 0.4, 0.8, 1.0]

finetune_params = {
    0.1: {'hidden_size': 1024, 'num_blocks': 10, 'learning_rate': 0.0001,  'num_frozen_blocks': 7},
    0.2: {'hidden_size': 1024, 'num_blocks': 19, 'learning_rate': 0.0001,  'num_frozen_blocks': 15},
    0.4: {'hidden_size': 1024, 'num_blocks': 13, 'learning_rate': 0.0001,  'num_frozen_blocks': 12},
    0.8: {'hidden_size': 1024, 'num_blocks': 20, 'learning_rate': 0.0001, 'num_frozen_blocks': 18},
    1.0: {'hidden_size': 1024, 'num_blocks': 11, 'learning_rate': 0.0001,  'num_frozen_blocks': 9}
}

foundation_model_dir = '/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/model/'
save_model_dir = '/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/model/'
data_dir = '/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/'

feature_sets = {
    #'Drop_ACSM': ['Temperature', 'RH', 'Bulk_DP', 'Num_Conc'],
    #'Drop_SD':   ['Temperature', 'RH', 'OA', 'SO4', 'Cl', 'NH4', 'NO3'],
    #'Drop_MET':  ['Bulk_DP', 'Num_Conc', 'OA', 'SO4', 'Cl', 'NH4', 'NO3']
	'Drop_N_':     ['Temperature', 'RH', 'Bulk_DP', 'Num_Conc', 'OA', 'SO4', 'Cl']
}

batch_sizes = [8, 16, 32]
num_epochs_list = [30, 40, 50]

results_all = []

for feature_set_name, features in feature_sets.items():
    for set_point in set_points:
        # ===== 数据路径与参数准备 =====
        foundation_path = os.path.join(foundation_model_dir, f'Foundation_Model_{set_point}.pth')
        train_path = os.path.join(data_dir, f'setpoint_{set_point}_train.csv')
        test_path  = os.path.join(data_dir, f'setpoint_{set_point}_test.csv')
        # 其余参数直接用 finetune_params[set_point]
        base_params = finetune_params[set_point].copy()
        num_blocks = base_params['num_blocks']
        hidden_size = base_params['hidden_size']
        learning_rate = base_params['learning_rate']

        # ========== 数据加载与标准化 ==========
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
        X_train = train_df[features]
        y_train = train_df.iloc[:, -1]
        X_test  = test_df[features]
        y_test  = test_df.iloc[:, -1]
        # 标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_np      = y_test.values

        for batch_size in batch_sizes:
            for num_epochs in num_epochs_list:
                # ========== DataLoader ==========
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # ========== 构建模型 ==========
                input_size = len(features)
                model = ResNet(input_size, num_blocks, hidden_size)
                # Foundation 权重加载（如有）
                if os.path.isfile(foundation_path):
                    model = load_partial_state_dict(model, foundation_path)
                # 只训练头尾层
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.fc_in.parameters():
                    param.requires_grad = True
                for param in model.fc_out.parameters():
                    param.requires_grad = True

                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=learning_rate
                )

                # ========== 训练 ==========
                best_mse = float('inf')
                best_state = None
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs.view(-1), y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        running_loss += loss.item() * X_batch.size(0)
                    epoch_loss = running_loss / len(train_loader.dataset)
                    if epoch_loss < best_mse:
                        best_mse = epoch_loss
                        best_state = model.state_dict()

                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor).view(-1).numpy()

                mse  = np.mean((y_pred - y_test_np) ** 2)
                rmse = np.sqrt(mse)
                r2   = r2_score(y_test_np, y_pred)
                mape = np.mean(np.abs((y_pred - y_test_np) / y_test_np)) * 100

                print(f"[{feature_set_name}] S={set_point} batch={batch_size} epoch={num_epochs}: "
                      f"MSE={mse:.2f} RMSE={rmse:.2f} R2={r2:.4f} MAPE={mape:.2f}")

                results_all.append({
                    'feature_set': feature_set_name,
                    'set_point': set_point,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape
                })

results_df = pd.DataFrame(results_all)
results_df.to_csv('finetune_feature_set_grid_results.csv', index=False)
print("All results saved as finetune_feature_set_grid_results.csv")
