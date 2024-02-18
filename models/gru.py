import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Added import
# Generate example time series data
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

scaler = MinMaxScaler(feature_range=(0, 1))



class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_window_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out




# Define the GRU model
class PredModel(nn.Module):
    def __init__(self, n_step, pred_window_len):
        super(PredModel, self).__init__()
        self.n_steps = n_step
        self.pred_window_len = pred_window_len
        self.n_features = 1

    def data_prepare(self, history_traffic):
        data_value = np.array(history_traffic).reshape(-1, 1)

        data_scaled = scaler.fit_transform(data_value)

        X, y = self.create_sequences(data_scaled)
        return X, y


    def create_sequences(self, data):  # n_step is the length of features
        X, y = [], []
        for i in range(len(data) - self.n_steps - self.pred_window_len):
            X.append(data[i:i + self.n_steps])
            print(len(data[i + self.n_steps:i + self.n_steps + self.pred_window_len]))
            y.append(data[i + self.n_steps:i + self.n_steps + self.pred_window_len])

        print(f'X:{X} y:{y}')
        return np.array(X), np.array(y)



    def train_test(self, X, y):

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = (
        #     train_test_split(X_tensor, y_tensor, train_size=0.8, test_size=0.2, shuffle=False))


        # 形成训练数据集
        train_data = TensorDataset(X_tensor, y_tensor)
        # test_data = TensorDataset(X_test, y_test)

        # 将数据加载成迭代器
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=8,
                                                   shuffle=True)

        # test_loader = torch.utils.data.DataLoader(test_data,
        #                                           batch_size=8,
        #                                           shuffle=False,
        #                                           drop_last=False)

        # Instantiate the model
        input_size = self.n_features
        hidden_size = 24
        output_size = self.pred_window_len

        model = GRUModel(input_size, hidden_size, output_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training the model
        num_epochs = 100
        running_loss = 0
        for epoch in range(num_epochs):
            train_bar = tqdm(train_loader)

            for data in train_bar:
                x_train_data, y_train_data = data
                optimizer.zero_grad()
                outputs = model(x_train_data)
                loss = criterion(outputs, y_train_data.reshape(-1, self.pred_window_len))
                # print(x_train.shape, y_train.shape, outputs.shape )
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}')
                running_loss = 0

        return scaler.inverse_transform(outputs.numpy())

        # #save model
        # torch.save(model.state_dict(), '../gru_model.pth')
        # Testing the model


        #
        # model.eval()
        # last_observation = X_test[-1].reshape(1, n_steps, 1)
        # test_loss = 0
        # with torch.no_grad():
        #     test_bar = tqdm(test_loader)
        #     print(test_bar)
        #     for data in test_bar:
        #         x_test_data, y_test_data = data
        #
        #         y_test_pred = model(x_test_data)
        #         test_loss += criterion(y_test_pred, y_test_data.reshape(-1, pred_window_len))
        #
        #     test_loss /= len(test_bar)
        #     print(f'Mean Squared Error on Test Data: {test_loss}')
        #     # next_day_prediction_inv = scaler.inverse_transform(y_test_pred.numpy())
        #     # print(f'Predicted user traffic for the next day: {next_day_prediction_inv[0, 0]:.2f}')
        #
        # y_test_pred = model(X_test)
        # predicted_values = scaler.inverse_transform(y_test_pred.detach().numpy())
        # print(f'{predicted_values.shape}')
        # true_values = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, self.pred_window_len))
        # plt.figure(figsize=(12, 8))
        # for t in range(1):
        #     plt.plot(list(range(t, t+self.pred_window_len)),predicted_values[t], "b")
        # # plt.plot(true_values, "r")
        # for t in range(1):
        #     plt.plot(list(range(t, t+self.pred_window_len)),true_values[t], "r")
        # plt.legend()
        # plt.show()
        #
        # true_test = y_test.detach().numpy().reshape(-1, self.pred_window_len)
        # pred_test = y_test_pred.detach().numpy()



    def __excute__(self, history_traffic):
        X, y = self.data_prepare(history_traffic)

        true_test, pred_test = self.train_test(X, y)
        # # true_test, pred_test = arima(X, y)
        mse = mean_squared_error(true_test, pred_test)
        mae = mean_absolute_error(true_test, pred_test)
        r2 = r2_score(true_test, pred_test)
        rmse = np.sqrt(mse)

        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'r2: {r2}')

#
# def arima(X, y):
#     # 定义训练数据和测试数据
#     train_size = int(len(y) * 0.8)
#     test_size = len(y) - train_size
#     train, test = y[:train_size], y[train_size:]
#     # 训练 ARIMA 模型
#     order = (1, 1, 1)  # ARIMA(p, d, q) 参数，这里是一个示例参数，你可能需要调整
#
#     # 预测未来1步
#     forecast_steps = 1
#     forecast_series = []
#
#     for i in range(test_size):
#         if i>=1:
#             dat = np.concatenate([train, test[:i]])
#         else:
#             dat = train
#         model = ARIMA(dat, order=order)
#         fit_model = model.fit()
#         forecast = fit_model.get_forecast(steps=forecast_steps)
#         forecast_series.append(forecast.predicted_mean)
#
#     # in_sample_pred = fit_model.predict()
#     # out_sample_pred = fit_model.predict(start=len(train)-1, end=len(train) + len(y)-train_size, \
#     #                                 dynamic=True)
#     plt.figure(figsize=(12, 8))
#     for t in range(len(test_size)):
#         plt.plot(list(range(t, t+pred_window_len)),forecast_series[i], "b")
#     plt.plot(test, "r")
#     plt.legend()
#     # plt.savefig('./gru.pdf')
#     return test, forecast_series
#
#
# def ACF(data):
#     # print(X)
#     df = pd.DataFrame({'full data': data})
#
#     plt.rcParams['axes.unicode_minus'] = False  # 修正坐标轴负号不显示的问题
#     fig = plt.figure(figsize=(10, 4), dpi=500)
#     ax = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     plot_acf(df['full data'].dropna().diff(1), lags=10, ax=ax)
#     plot_pacf(df['full data'].dropna().diff(1), lags=15, ax=ax2)
#     plt.show()


