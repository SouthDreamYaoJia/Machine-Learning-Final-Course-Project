from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_step = 96
num_features = 13
prediction_size = 240

# Load datasets
train_data = pd.read_csv('Dataset/test_data.csv')
test_data = pd.read_csv('Dataset/train_data.csv')

# Data preprocessing: Modify index and remove unnecessary columns
def preprocess_data(data):
    data['dteday'] = pd.to_datetime(data['dteday'])
    data['datetime'] = data['dteday'] + pd.to_timedelta(data['hr'], unit='h')
    data.set_index('datetime', inplace=True)
    data.drop(['dteday', 'instant', 'casual', 'registered'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Normalize the target column 'cnt'
def normalize_data(data, scaler):
    data['cnt'] = scaler.fit_transform(data['cnt'].values.reshape(-1, 1))
    return data

scaler = MinMaxScaler(feature_range=(0, 1))
train_data = normalize_data(train_data, scaler)
test_data = normalize_data(test_data, scaler)

# Create data windows
def create_data_window(data, window_size, prediction_size):
    dataX, dataY = [], []
    for i in range(len(data) - window_size - prediction_size + 1):
        a = data.iloc[i:(i + window_size), :].values
        dataX.append(a)
        dataY.append(data.iloc[i + window_size:i + window_size + prediction_size, -1].values)
    return np.array(dataX), np.array(dataY)

X_train, y_train = create_data_window(train_data, time_step, prediction_size)
X_test, y_test = create_data_window(test_data, time_step, prediction_size)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, time_step, num_features, dropout_rate=0.2, prediction_size=240):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(50, 64)
        self.relu1 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, prediction_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out[:, -1, :])
        out = self.relu1(self.fc1(out))
        out = self.dropout4(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout5(out)
        out = self.fc3(out)
        return out
# 进行五轮训练
num_experiments = 5
train_mses = []
train_maes = []
test_mses = []
test_maes = []

best_model = None
best_test_mse = float('inf')
best_experiment = -1

for i in range(num_experiments):
    best_lr = 0.001100

    # 创建模型实例
    model = LSTMModel(time_step, num_features, prediction_size=prediction_size).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    
    # 训练模型
    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # 预测
    model.eval()
    with torch.no_grad():
        train_predict = []
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            train_predict.append(output.cpu().numpy())
        train_predict = np.concatenate(train_predict, axis=0)
        
        test_predict = []
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            test_predict.append(output.cpu().numpy())
        test_predict = np.concatenate(test_predict, axis=0)
    
    # 计算MSE和MAE（使用归一化后的值）
    train_mse = mean_squared_error(y_train.cpu().numpy(), train_predict)
    train_mae = mean_absolute_error(y_train.cpu().numpy(), train_predict)
    test_mse = mean_squared_error(y_test.cpu().numpy(), test_predict)
    test_mae = mean_absolute_error(y_test.cpu().numpy(), test_predict)
    
    # 记录结果
    train_mses.append(train_mse)
    train_maes.append(train_mae)
    test_mses.append(test_mse)
    test_maes.append(test_mae)
    
    print(f'Experiment {i + 1} - Train MSE: {train_mse}, Train MAE: {train_mae}')
    print(f'Experiment {i + 1} - Test MSE: {test_mse}, Test MAE: {test_mae}')

    # 保存最佳模型
    if test_mse < best_test_mse:
        best_test_mse = test_mse
        best_model = model
        best_experiment = i

    # 反向转换预测值和实际值
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1)).reshape(train_predict.shape)
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1)).reshape(test_predict.shape)
    y_train_inv = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).reshape(y_train.shape)
    y_test_inv = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).reshape(y_test.shape)

# 保存最佳模型
torch.save(best_model.state_dict(), f'Result_BestModle/LSTM_240_best_model_experiment_{best_experiment + 1}.pth')

# 计算平均值和标准差
train_mse_avg = np.mean(train_mses)
train_mse_std = np.std(train_mses)
train_mae_avg = np.mean(train_maes)
train_mae_std = np.std(train_maes)
test_mse_avg = np.mean(test_mses)
test_mse_std = np.std(test_mses)
test_mae_avg = np.mean(test_maes)
test_mae_std = np.std(test_maes)

print(f'Average Train MSE: {train_mse_avg}, Std: {train_mse_std}')
print(f'Average Train MAE: {train_mae_avg}, Std: {train_mae_std}')
print(f'Average Test MSE: {test_mse_avg}, Std: {test_mse_std}')
print(f'Average Test MAE: {test_mae_avg}, Std: {test_mae_std}')

# 写入文件
with open(f'Result_Record/实验结果_LSTM_240_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.txt', 'w') as f:
    f.write(f'Best Experiment: {best_experiment + 1}\n')
    f.write(f'Best Test MSE: {best_test_mse}\n')
    f.write(f'Average Train MSE: {train_mse_avg}, Std: {train_mse_std}\n')
    f.write(f'Average Train MAE: {train_mae_avg}, Std: {train_mae_std}\n')
    f.write(f'Average Test MSE: {test_mse_avg}, Std: {test_mse_std}\n')
    f.write(f'Average Test MAE: {test_mae_avg}, Std: {test_mae_std}\n')


def plot_results(real_values, predictions, title, num_samples=500, name='result.png'):
    """
    绘制真实值和预测值的折线图。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(real_values[:num_samples], label="Real Values", color="blue", alpha=0.7)
    plt.plot(predictions[:num_samples], label="Predictions", color="red", linestyle="--", alpha=0.7)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(name)

# 获取时间字符窜
timeStamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

# 绘制训练集结果
plot_results(
    y_train_inv.flatten(),
    train_predict.flatten(),
    title="Training Set: Real vs Predicted",
    num_samples=500,  # 可调整绘制样本数量
    name = 'Result_fig/LSTM_240_f_1_MultiValue_train_prediction_'+timeStamp+'.png'
)

# 绘制测试集结果
plot_results(
    y_test_inv.flatten(),
    test_predict.flatten(),
    title="Test Set: Real vs Predicted",
    num_samples=500,  # 可调整绘制样本数量
    name = 'Result_fig/LSTM_240_f_1_MultiValue_test_prediction_'+timeStamp+'.png'
)