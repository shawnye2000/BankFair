import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler(feature_range=(0, 1))



class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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
        print(f'his tra:{history_traffic}')
        data_value = np.array(history_traffic).reshape(-1, 1)

        data_scaled = scaler.fit_transform(data_value)

        print(f'data sacled:{data_scaled}')
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
        print(X,y)
        # x_test =
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        X_test = torch.cat((X_tensor[-1][self.pred_window_len:], y_tensor[-1]), dim=0)
        # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = (
        #     train_test_split(X_tensor, y_tensor, train_size=1, test_size=0, shuffle=False))

        # 形成训练数据集
        train_data = TensorDataset(X_tensor, y_tensor)

        # 将数据加载成迭代器
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   shuffle=False)
        # Instantiate the model
        input_size = self.n_features
        hidden_size = 24
        output_size = self.pred_window_len

        model = GRUModel(input_size, hidden_size, output_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

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

        with torch.no_grad():
            forecast = model(X_test.unsqueeze(0))
        forecast = scaler.inverse_transform(forecast.numpy())
        print(f'forecast:{forecast}')
        return forecast[0]




    def excute(self, history_traffic):
        X, y = self.data_prepare(history_traffic)

        pred_result = self.train_test(X, y)
        return pred_result

