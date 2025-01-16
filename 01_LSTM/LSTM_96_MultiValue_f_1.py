from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 参数设置
time_step = 96
num_features = 13  # 数据特征数
num_outputs = 96  # 输出时间步数

# 数据加载和预处理
train_data = pd.read_csv('Dataset/test_data.csv')
test_data = pd.read_csv('Dataset/train_data.csv')

# 数据预处理函数
def DatasetDeal_1(data):
    data['dteday'] = pd.to_datetime(data['dteday'])
    data['datetime'] = data['dteday'] + pd.to_timedelta(data['hr'], unit='h')
    data.set_index('datetime', inplace=True)
    data.drop(['dteday', 'instant', 'casual', 'registered'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data

train_data = DatasetDeal_1(train_data)
test_data = DatasetDeal_1(test_data)

def DatsScaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['cnt'].values.reshape(-1, 1))
    data['cnt'] = scaled_data
    return scaler

scaler = DatsScaler(train_data)
DatsScaler(test_data)

# 窗口函数（生成多步预测标签）
def CreateDataWindow_all(data, window_size=96, output_steps=96):
    dataX, dataY = [], []
    for i in range(len(data) - window_size - output_steps):
        a = data.iloc[i:(i + window_size), :].values  # 输入窗口
        b = data.iloc[(i + window_size):(i + window_size + output_steps), -1].values  # 输出窗口
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

X_train, y_train = CreateDataWindow_all(train_data, time_step, num_outputs)
X_test, y_test = CreateDataWindow_all(test_data, time_step, num_outputs)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, time_step, num_features, num_outputs, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(50, num_outputs)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])
        out = self.fc(out)
        return out

# 进行5轮实验
num_experiments = 5
train_mses = []
train_maes = []
test_mses = []
test_maes = []

best_model = None
best_test_mse = float('inf')
best_experiment = -1

for experiment in range(num_experiments):
    print(f'Experiment {experiment + 1}/{num_experiments}')
    
    # 训练和评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(time_step, num_features, num_outputs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    train_mses.append(train_mse)
    train_maes.append(train_mae)
    test_mses.append(test_mse)
    test_maes.append(test_mae)

    print(f'Experiment {experiment + 1} - Train MSE: {train_mse}, Train MAE: {train_mae}')
    print(f'Experiment {experiment + 1} - Test MSE: {test_mse}, Test MAE: {test_mae}')

    # 保存最佳模型
    if test_mse < best_test_mse:
        best_test_mse = test_mse
        best_model = model
        best_experiment = experiment

    # 反向转换预测值和实际值
    train_actual = scaler.inverse_transform(y_train.cpu().numpy())
    test_actual = scaler.inverse_transform(y_test.cpu().numpy())
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

# 保存最佳模型
torch.save(best_model.state_dict(), f'Result_BestModle/LSTM_96_best_model_experiment_{best_experiment + 1}.pth')


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
with open(f'Result_Record/实验结果_LSTM_96_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.txt', 'w') as f:
    f.write(f'Best Experiment: {best_experiment + 1}\n')
    f.write(f'Best Test MSE: {best_test_mse}\n')
    f.write(f'Average Train MSE: {train_mse_avg}, Std: {train_mse_std}\n')
    f.write(f'Average Train MAE: {train_mae_avg}, Std: {train_mae_std}\n')
    f.write(f'Average Test MSE: {test_mse_avg}, Std: {test_mse_std}\n')
    f.write(f'Average Test MAE: {test_mae_avg}, Std: {test_mae_std}\n')


# 绘制结果
timeStamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

plt.figure(figsize=(12, 6))
plt.plot(train_actual[-1], label='Actual Train Count', color='blue')
plt.plot(train_predict[-1], label='Predicted Train Count', color='red')
plt.title('Training Data Prediction')
plt.xlabel('Step')
plt.ylabel('Count')
plt.legend()
plt.savefig(f'Result_fig/LSTM_96_f_1_MultiValue_train_prediction_experiment_{experiment + 1}_{timeStamp}.png')

plt.figure(figsize=(12, 6))
plt.plot(test_actual[-1], label='Actual Test Count', color='blue')
plt.plot(test_predict[-1], label='Predicted Test Count', color='green')
plt.title('Testing Data Prediction')
plt.xlabel('Step')
plt.ylabel('Count')
plt.legend()
plt.savefig(f'Result_fig/LSTM_96_f_1_MultiValue_test_prediction_experiment_{experiment + 1}_{timeStamp}.png')
