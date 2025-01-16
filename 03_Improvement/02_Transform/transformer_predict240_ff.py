import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fftpack import fft


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 数据加载函数
def load_data(train_file, test_file):
    # 读取 CSV 文件
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # 假设 'cnt' 是单车租赁数量
    train_cnt = train_data['cnt'].values
    test_cnt = test_data['cnt'].values

    return train_cnt, test_cnt


# 数据处理函数，加入傅里叶变换特征
def create_dataset_with_fft(data, time_step=96, output_step=240, fft_components=48):
    X, y = [], []
    for i in range(len(data) - time_step - output_step):
        # 原始时间序列
        time_series_segment = data[i:(i + time_step)]
        # 傅里叶变换提取频域特征
        fft_features = np.abs(fft(time_series_segment))[:fft_components]
        # 将时域特征与频域特征拼接
        combined_features = np.concatenate([time_series_segment, fft_features])
        X.append(combined_features)
        y.append(data[(i + time_step):(i + time_step + output_step)])
    return np.array(X), np.array(y)


# 修改后的 Transformer 模型，适配扩展后的输入
class TransformerModelWithFFT(nn.Module):
    def __init__(self, input_size, output_size, fft_components=48, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 hidden_dim=512):
        super(TransformerModelWithFFT, self).__init__()
        self.input_size = input_size + fft_components  # 输入维度增加
        self.output_size = output_size

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=nhead,
                                                        dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.input_size, nhead=nhead,
                                                        dim_feedforward=hidden_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(self.input_size, output_size)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc_out(output)
        return output


# 评估模型性能并显示预测结果
def evaluate_model(model, test_loader, scaler, cishu):
    model.eval()
    y_true = []
    y_pred = []
    predictions_list = []  # 用于保存预测结果

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch, X_batch)  # 预测时使用输入数据

            if np.isnan(y_batch.cpu().numpy()).any() or np.isnan(predictions.cpu().numpy()).any():
                print("NaN detected in predictions or true values.")
                continue  # 跳过该批次

            y_true.append(y_batch)
            y_pred.append(predictions)
            predictions_list.append(predictions.cpu().numpy())  # 保存预测结果

    # 拼接所有预测和真实值
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # 将标准化的预测值还原为原始值
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()  # 逆标准化真实值并展平为一维
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()  # 逆标准化预测值并展平为一维

    # 将负值的纵坐标设置为 0
    # y_pred = np.maximum(y_pred, 0)  # 将 y_pred 中的负值设置为 0

    # 只取前240个小时的数据
    y_true = y_true[:240]
    y_pred = y_pred[:240]

    # 绘制真实值和预测值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(range(240), y_true, label='Actual', color='blue', alpha=0.6)  # 横坐标为小时，范围0-239
    plt.plot(range(240), y_pred, label='Predicted', color='red', alpha=0.6)  # 横坐标为小时，范围0-239
    plt.title('Actual vs Predicted Bike Counts (240 Hours)')
    plt.xlabel('Hour (0 to 239)')
    plt.ylabel('Bike Count (cnt)')
    plt.legend()
    filename = f'actual_vs_predicted_bike_counts_240_hours_ff_{cishu + 1}_48.png'
    plt.savefig(filename)  # 保存为图片文件
    plt.close()

    return mse, mae, predictions_list


# 训练和测试模型的函数
def train_and_evaluate_with_fft(train_cnt, test_cnt, time_step=96, output_step=240, fft_components=48, num_epochs=3,
                                num_experiments=5, batch_size=32):
    # 标准化数据
    scaler = MinMaxScaler()
    train_cnt_scaled = scaler.fit_transform(train_cnt.reshape(-1, 1)).ravel()  # 保持一维
    test_cnt_scaled = scaler.transform(test_cnt.reshape(-1, 1)).ravel()  # 保持一维

    # 创建包含 FFT 特征的训练数据集和测试数据集
    X_train, y_train = create_dataset_with_fft(train_cnt_scaled, time_step, output_step, fft_components)
    X_test, y_test = create_dataset_with_fft(test_cnt_scaled, time_step, output_step, fft_components)

    # 创建 PyTorch 数据集
    train_dataset = TimeSeriesDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TimeSeriesDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_mse = float('inf')  # 初始最优模型的 MSE 设置为无限大
    best_mae = float('inf')  # 初始最优模型的 MAE 设置为无限大
    best_model = None

    mse_scores = []
    mae_scores = []
    all_predictions = []  # 保存所有的预测结果

    # 进行多次实验
    for exp in range(num_experiments):
        # 初始化模型，损失函数和优化器
        model = TransformerModelWithFFT(input_size=time_step, output_size=output_step, fft_components=fft_components)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # 训练模型
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch, X_batch)  # 使用相同的数据进行训练
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(
                f'Experiment {exp + 1}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

        # 在测试集上评估
        mse, mae, predictions_list = evaluate_model(model, test_loader, scaler, exp)
        mse_scores.append(mse)
        mae_scores.append(mae)
        all_predictions.append(predictions_list)  # 保存预测结果

        # 如果当前模型表现更好，保存模型
        if mse < best_mse and mae < best_mae:
            best_mse = mse
            best_mae = mae
            best_model = model
            torch.save(model.state_dict(), 'best_model_240_ff_48.pth')  # 保存最优模型
            # 保存模型的状态字典，同时保存 mse_scores 和 mae_scores
            model_info = {
                'model_state_dict': model.state_dict(),
                'mse_scores': mse_scores,
                'mae_scores': mae_scores,
            }
            torch.save(model_info, 'best_model_with_scores_240_ff_48.pth')  # 保存最优模型和相关指标

    # 计算均值和标准差
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)

    print(f'\nAverage MSE: {mse_mean:.4f} ± {mse_std:.4f}')
    print(f'Average MAE: {mae_mean:.4f} ± {mae_std:.4f}')

    return mse_mean, mse_std, mae_mean, mae_std


# 加载最优模型
def load_best_model(model_class, model_path='best_model_240_ff_48.pth'):
    model = model_class(input_size=96, output_size=240)  # 初始化模型
    model.load_state_dict(torch.load(model_path))  # 加载最优模型的权重
    model.eval()  # 设置模型为评估模式
    return model


# 加载最优模型和指标
def load_best_model_with_scores(model_class, model_path='best_model_with_scores_240_ff_48.pth'):
    # 加载模型的状态字典和相关指标
    checkpoint = torch.load(model_path)

    model = model_class(input_size=96, output_size=240)  # 初始化模型
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载最优模型的权重
    model.eval()  # 设置模型为评估模式

    mse_scores = checkpoint['mse_scores']
    mae_scores = checkpoint['mae_scores']

    return model, mse_scores, mae_scores


# 评估并绘制预测结果
def plot_predictions(model, test_loader, scaler):
    model.eval()  # 确保模型是评估模式
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch, X_batch)  # 进行预测

            y_true.append(y_batch)
            y_pred.append(predictions)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    # 将标准化的预测值还原为原始值
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()  # 逆标准化真实值并展平为一维
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()  # 逆标准化预测值并展平为一维

    # 将负值的纵坐标设置为 0
    # y_pred = np.maximum(y_pred, 0)  # 将 y_pred 中的负值设置为 0

    # 只取前240个小时的数据
    y_true = y_true[:240]
    y_pred = y_pred[:240]

    # 绘制真实值和预测值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(range(240), y_true, label='Actual', color='blue', alpha=0.6)  # 横坐标为小时，范围0-239
    plt.plot(range(240), y_pred, label='Predicted', color='red', alpha=0.6)  # 横坐标为小时，范围0-239
    plt.title('Actual vs Predicted Bike Counts (240 Hours)')
    plt.xlabel('Hour (0 to 239)')
    plt.ylabel('Bike Count (cnt)')
    plt.legend()
    plt.savefig('best_actual_vs_predicted_bike_counts_240_hours_ff_48.png')  # 保存为图片文件
    plt.close()

    best_model, mse_scores, mae_scores = load_best_model_with_scores(TransformerModelWithFFT)
    print("MSE Scores:", mse_scores[-1])
    print("MAE Scores:", mae_scores[-1])


# 主函数
if __name__ == "__main__":
    # 加载数据
    train_cnt, test_cnt = load_data('train_data1.csv', 'test_data1.csv')
    train_cnt = train_cnt.astype(np.float32)
    test_cnt = test_cnt.astype(np.float32)

    # 获取输入并赋值
    sel = int(input("Training(0) OR Observing the best model(1): "))  # 获取用户输入的字符串

    if sel == 0:
        # 进行训练和评估
        mse_mean, mse_std, mae_mean, mae_std = train_and_evaluate_with_fft(train_cnt, test_cnt)

    else:
        # 数据预处理
        scaler = MinMaxScaler()
        train_cnt_scaled = scaler.fit_transform(train_cnt.reshape(-1, 1)).ravel()  # 标准化训练数据
        test_cnt_scaled = scaler.transform(test_cnt.reshape(-1, 1)).ravel()  # 使用训练集的参数标准化测试数据

        X_test, y_test = create_dataset_with_fft(test_cnt_scaled, time_step=96, output_step=240, fft_components=48)

        test_dataset = TimeSeriesDataset(torch.tensor(X_test, dtype=torch.float32),
                                         torch.tensor(y_test, dtype=torch.float32))

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 加载最优模型
        model = load_best_model(TransformerModelWithFFT)

        # 绘制预测结果并还原数据
        plot_predictions(model, test_loader, scaler)
