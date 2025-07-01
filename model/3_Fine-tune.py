import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import random

# ------------------------------------------
# (1) 设置随机种子以保证可复现
# ------------------------------------------
def set_seed(seed):
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

# ------------------------------------------
# (2) 定义 ResNet 结构（同你给出的代码）
# ------------------------------------------
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

batch_sizes = [8, 16, 32]
num_epochs_list = [30, 40, 50]
features = ['Temperature', 'RH', 'Bulk_DP', 'Num_Conc', 'OA', 'SO4', 'Cl', 'NH4', 'NO3']
partmc_train_path = '/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/partmc_train.csv'
partmc_train_data = pd.read_csv(partmc_train_path)
X_partmc = partmc_train_data[features].values
scaler_X = StandardScaler()
scaler_X.fit(X_partmc)

finetune_params = {
    0.1: {'hidden_size': 1024, 'num_blocks': 10, 'learning_rate': 0.000530435797934885,  'num_frozen_blocks': 7,  'weight_decay':  0.008497148468566256},
    0.2: {'hidden_size': 1024, 'num_blocks': 19, 'learning_rate': 0.0009907214001954557,  'num_frozen_blocks': 15, 'weight_decay': 0.0008450232969616185},
    0.4: {'hidden_size': 1024, 'num_blocks': 13, 'learning_rate': 0.0008399151983819402,  'num_frozen_blocks': 12, 'weight_decay': 0.0012652607042064328},
    0.8: {'hidden_size': 1024, 'num_blocks': 20, 'learning_rate': 0.0009661785734282122, 'num_frozen_blocks': 18, 'weight_decay':  0.0005279042637155721},
    1.0: {'hidden_size': 1024, 'num_blocks': 11, 'learning_rate': 0.0005540192611872995,  'num_frozen_blocks': 9,  'weight_decay': 0.0009568555327884106}
}
set_points = list(finetune_params.keys())

results_all = []
for sp in set_points:
    print(f"\n==== 处理 set_point = {sp} ====")
    # 加载数据
    train_path = f'/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/setpoint_{sp}_train.csv'
    test_path  = f'/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/data/setpoint_{sp}_test.csv'
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        print(f"数据文件不存在，跳过 {sp}")
        continue

    arm_train = pd.read_csv(train_path)
    arm_test  = pd.read_csv(test_path)
    X_train = scaler_X.transform(arm_train[features].values)
    y_train = arm_train['N_CCN'].values
    X_test = scaler_X.transform(arm_test[features].values)
    y_test  = arm_test['N_CCN'].values

    # 其余参数保持finetune_params设定
    params = finetune_params[sp]
    hidden_size = params['hidden_size']
    num_blocks = params['num_blocks']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    num_frozen_blocks = params['num_frozen_blocks']

    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model = ResNet(in_features=len(features), num_blocks=num_blocks, hidden_size=hidden_size)
            # 若有 Foundation 权重可加载
            foundation_path = f'/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/model/Foundation_Model_{sp}.pth'
            if os.path.isfile(foundation_path):
                model.load_state_dict(torch.load(foundation_path, map_location='cpu'))

            for idx, block in enumerate(model.blocks):
                if idx < num_frozen_blocks:
                    for p in block.parameters():
                        p.requires_grad = False

            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            model.train()
            for epoch in range(num_epochs):
                for X_b, y_b in train_loader:
                    optimizer.zero_grad()
                    out = model(X_b).view(-1)
                    loss = criterion(out, y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_pred = model(X_test_tensor).view(-1).numpy()

            mse  = np.mean((y_pred - y_test) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

            print(f"set_point={sp}, batch_size={batch_size}, num_epochs={num_epochs}  ==> "
                  f"MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.1f}%")

            results_all.append({
                'set_point': sp,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            })

# 保存所有结果
results_df = pd.DataFrame(results_all)
results_df.to_csv('/data/keeling/a/xx24/e/proj_ml/code_ml_surfactant_ccn/model/fine_tune_grid_results.csv', index=False)
print(f"\n结果已保存到 fine_tune_grid_results.csv")