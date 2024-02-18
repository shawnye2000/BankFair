import os

import torch.nn as nn
import numpy as np
import argparse
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import Bankruptcy, pct, reg_exp, TFROM, oracle, FairRec, FairRecPlus, CPFair, min_regularizer, k_neighbor, Welf, ROAP, P_MMF, popular
from recbole.quick_start.quick_start import run_recbole
import math
import bankrupt as bkr
from random import choice
gpu_list = ['0','1','2','3']
os.environ["CUDA_VISIBLE_DEVICES"] = choice(gpu_list)

'''
TODO:
- create a simulator: which use full time interaction as a input
- use recbole-master to train the 10 days data to predict the next day's user preference
- split the train and test 
- use former 80% dataset as an train as 20% as test
- shuffle the test and valid
'''
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
###


def MMF(provider_exposure):
    return min(provider_exposure)


def Accuracy(UI_matrix, recommend_items):
    if isinstance(UI_matrix, torch.Tensor):
        UI_matrix = UI_matrix.cpu().numpy()
    if isinstance(recommend_items, torch.Tensor):
        recommend_items = recommend_items.cpu().numpy()
    UI_values = UI_matrix[recommend_items]
    accuracy = np.sum(UI_values)
    return accuracy


def NDCG(one_UI, topk_items):
    if isinstance(one_UI, torch.Tensor):
        one_UI = one_UI.cpu().numpy()
    nor_dcg = 0
    # simulator_sort = np.sort(simulator[upcoming_user_list].reshape(user_number, -1), axis=-1)
    UI_matrix_sort = np.sort(one_UI, axis=-1)
    dcg = 0
    x_recommended = topk_items
    for k in range(args.topk):
        nor_dcg = nor_dcg + UI_matrix_sort[dataset.item_num - k - 1] / np.log2(k + 2)  # ground truth
        dcg = dcg + one_UI[x_recommended[k]] / np.log2(k + 2)  # real recommend

    # print(f'dcg:{dcg} nor_dcg[i]:{nor_dcg[i]}')
    return dcg / nor_dcg

def metrics(topk_items, true_item):
    total_hitrate = 0
    recall = 0
    dcg = 0
    idcg = 0
    for no, iid in enumerate(topk_items):
        if iid == true_item:
            recall += 1
            dcg += 1.0 / math.log(no + 2, 2)
            break

    idcg += 1.0

    total_recall = recall / 1
    total_ndcg = dcg / idcg
    if recall > 0:
        total_hitrate = 1
    return total_ndcg, total_recall, total_hitrate

def metrics2(topk_items, true_items):
    total_hitrate = 0
    recall = 0
    dcg = 0
    idcg = 0
    # print(f'true items:{true_items}')
    for no, iid in enumerate(topk_items):
        if iid in true_items:
            recall += 1
            dcg += 1.0 / math.log(no + 2, 2)

    for no in range(len(true_items)):
        idcg += 1.0 / math.log(no + 2, 2)


    total_recall = recall / len(true_items)
    total_ndcg = dcg / idcg

    if recall > 0:
        total_hitrate = 1
    return total_ndcg, total_recall, total_hitrate



def Gini(provider_exposure):
    exposure = sorted(provider_exposure, reverse=False)
    exposure = np.array(exposure)
    total = 0
    for i, xi in enumerate(exposure[:-1], 1):
        total += np.sum(np.abs(xi-exposure[i:]))
    return total / (len(exposure)**2 * np.mean(exposure))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def make_date_list():
    # 指定起始日期和结束日期
    date_format = '%Y-%m-%d'
    start_date = datetime.strptime(args.start_date, date_format)
    end_date = datetime.strptime(args.end_date, date_format)


    # 创建一个空列表来存储日期字符串
    date_list = []

    # 生成日期范围并添加到列表中
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return date_list



class DatasetStats:
    def __init__(self):
        self.user_num = 0
        self.item_num = 0
        self.provider_num = 0
        self.UI_matrix = None
        self.A = None
        self.item2provider = {}
        self.provider_item_counts = None
        self.provideLen = None  # provider interaction times
        self.inter_dataframe = None
        self.uid_field = ''
        self.iid_field = ''
        self.label_field = ''
        self.time_field = ''
        self.provider_field = ''

    def load_from_file(self, file_path):
        # self.UI_matrix = np.load(os.path.join(file_path, args.dataset, args.dataset + '.npy'))
        # self.UI_matrix = sigmoid(self.UI_matrix)


        processed_data_path = os.path.join(file_path, args.dataset, args.dataset + '.inter')
        self.inter_dataframe = pd.read_csv(processed_data_path, sep='\t')
        # table head:review_id:token	user_id:token	business_id:token	stars:float	useful:float funny:float	cool:float	date:float

        self.uid_field, self.iid_field, self.label_field, self.time_field, self.provider_field = self.inter_dataframe.columns

        self.inter_dataframe['interaction_time'] = pd.to_datetime(self.inter_dataframe[self.time_field], unit='ms' if args.dataset=='KuaiRand-1K' else 's')
        self.inter_dataframe = self.inter_dataframe.sort_values(by='interaction_time', ascending=True) # 递增
        self.provider_num = len(self.inter_dataframe[self.provider_field].unique())
        self.user_num, self.item_num = len(self.inter_dataframe[self.uid_field].unique()),len(self.inter_dataframe[self.iid_field].unique())
        print(f'user_num {self.user_num} item_num {self.item_num}')
        self.providerLen = np.array(self.inter_dataframe.groupby(self.provider_field).size().values)
        print(f'providerLen:{self.providerLen}')


        self.rho = (1 - args.alpha) * self.providerLen / np.sum(self.providerLen) + args.alpha * np.array(
            [2 / self.provider_num for i in range(self.provider_num)])
        print(f'rho:{self.rho}')
        tmp = self.inter_dataframe[[self.iid_field, self.provider_field]].drop_duplicates()
        unique_item_datas = self.inter_dataframe.drop_duplicates(subset=[self.iid_field])
        self.provider_item_counts = np.array(unique_item_datas.groupby(self.provider_field)[self.iid_field].count())
        print(f'provider-item-count:{self.provider_item_counts}')
        self.item2provider = {x: y for x, y in zip(tmp[self.iid_field], tmp[self.provider_field])}
        # if args.dataset == 'KuaiRec':
        #     self.item2provider[1225] = 0
        # A is item-provider matrix
        A = np.zeros((self.item_num, self.provider_num))
        iid2pid = []
        for i in range(self.item_num):

            iid2pid.append(self.item2provider[i])
            A[i, self.item2provider[i]] = 1
        self.A = np.array(A)
        print(f'item-provider adjacency matrix{self.A.shape}')


def choose_model(dataset):
    if args.fairness_model == 'Bankruptcy':
        fairness_model = Bankruptcy.Bankruptcy(rho=dataset.rho, M=dataset.A, TopK=args.topk, item_num=dataset.item_num,
                                               provider_item_counts=dataset.provider_item_counts,
                                               base_model=args.base_model,
                                               theta=args.theta,
                                               fairness_model=args.fairness_model)
    else:
        raise NotImplementedError("Not supported model names")
    return fairness_model


def get_train_ui_matrix(dataset, train_df, date, ori_train_ui_matrix):
    # get 10 days data before date
    date = pd.to_datetime(date).date()
    train_df = train_df[[dataset.uid_field, dataset.iid_field, dataset.label_field, dataset.time_field]]


    temp_df_save_path = os.path.join('temp_data', 'inter', args.dataset)
    npy_path = os.path.join('temp_data', 'npy', args.dataset)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(temp_df_save_path):
        os.makedirs(temp_df_save_path)

    npy_name = args.base_model + '_' + args.dataset + '_' + f'top{args.topk}' + '_' + date.strftime("%Y-%m-%d") + '.npy'
    if os.path.exists(os.path.join(npy_path, npy_name)):
        train_ui_matrix = np.load(os.path.join(npy_path, npy_name))

    else:

        users_ori = train_df[dataset.uid_field].unique().astype(str).tolist()
        items_ori = train_df[dataset.iid_field].unique().astype(str).tolist()
        train_df.to_csv(os.path.join(temp_df_save_path, args.dataset + '.inter'), sep='\t', index=False)

        config_file_list = args.config_files.strip().split(' ') if args.config_files else None
        train_result = run_recbole(model=args.base_model, dataset=args.dataset,
                                   config_file_list=config_file_list,
                                   big_ui_matrix=ori_train_ui_matrix,
                                   users_ori=users_ori,
                                   items_ori=items_ori)
        train_ui_matrix = train_result['ui_matrix']  # update the train ui matrix
        print(f'test result:{train_result["test_result"]}')
        np.save(os.path.join(npy_path, npy_name), train_ui_matrix)  # save npy
    return train_ui_matrix




def min_max_normalize(matrix):
    min_vals = matrix.min(axis=1, keepdims=True)
    max_vals = matrix.max(axis=1, keepdims=True)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    return normalized_matrix



class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out


    def user_traffic_prediction(history_traffic):
        input_size = 1  # n_features
        hidden_size = 24
        output_size = 7
        n_step = 30



        data_np = np.array(history_traffic).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_scaled = scaler.fit_transform(data_np)

        X_data = data_scaled[-n_step:]
        X_data = torch.tensor(X_data, dtype=torch.float32).view(1, -1, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_data[:-1])
            loss = criterion(outputs, X_data[1:])
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


        loaded_model = GRUModel(input_size, hidden_size, output_size)
        loaded_model.load_state_dict(torch.load('gru_model.pth'))
        loaded_model.eval()

        predict_data = loaded_model(X_data)

        return scaler.inverse_transform(predict_data.detach().numpy())



def run_baseline(dataset, model):
    # train_test_split_date = args.start_date
    sum_provider_item_counts = sum(dataset.provider_item_counts)
    print(f'sum provider item counts:{sum_provider_item_counts}')

    inter_data = dataset.inter_dataframe

    train_ui_matrix = np.zeros((dataset.user_num, dataset.item_num),  dtype=float)
    column_names = ['userid:token', 'interaction_time',
                    'NDCG_sui',
                   'recommend_list']
    result_df = pd.DataFrame(columns=column_names)
    if args.fairness_model=='Bankruptcy':
        df_filename = ('./df/run_baseline4' + args.fairness_model + '_'+args.dataset+ '_'+
                       args.base_model + f'_interval{args.interval_len}_top{args.topk}_estate{args.total_estate}_theta{args.theta}'+'.csv')
    result_df.to_csv(df_filename, index=False, header=True)

    train_test_split_date = args.start_date
    # change here
    train_df = inter_data[inter_data['interaction_time'].dt.date <
                          pd.to_datetime(train_test_split_date).date()]
    print(f'train len:{len(train_df)}')
    test_df = inter_data[inter_data['interaction_time'].dt.date >=
                         pd.to_datetime(train_test_split_date).date()]
    print(f'test len:{len(test_df)}')
    test_df = test_df[test_df[dataset.label_field] == 1]

    train_ui_matrix = get_train_ui_matrix(dataset,
                                          train_df,
                                          date=train_test_split_date,
                                          ori_train_ui_matrix=train_ui_matrix)
    sig_train_ui_matrix = min_max_normalize(train_ui_matrix)

    upcoming_user_list = test_df[[dataset.uid_field,
                                  'interaction_time',
                                  dataset.provider_field,
                                  dataset.iid_field,
                                  dataset.label_field]]
    upcoming_user_list.set_index('interaction_time', inplace=True)

    test_providerLen = np.array(upcoming_user_list.groupby(dataset.provider_field).size().values)
    print(f'test provider len: {test_providerLen}')
    time_interval_len = args.interval_len + 'H'
    result_dfs = [group for _, group in upcoming_user_list.resample(time_interval_len)]


    user_number = len(upcoming_user_list)
    print(f'{user_number} users arrives')

    beta = np.zeros(dataset.provider_num)


    for day_idx, subset_df in tqdm(enumerate(result_dfs), desc="user arriving", unit="user"):

        if args.fairness_model == 'Bankruptcy':
            # update the estate
            if day_idx == 0:
                total_estate = np.array([args.total_estate for i in range(dataset.provider_num)])
            else:
                print(f'beta:{beta}')
                print(f'daily_rho:{daily_rho[:,0]}')
                total_estate = np.clip(total_estate - daily_rho[:, 0] + beta.cpu().numpy(), a_min=0.0, a_max=None)


            # user traffic prediction
            # we use gru to predict the future user traffic
            # history traffic =  [len(i) for i in result_dfs[:day_idx]]


            print(f'total estate:{total_estate}')
            daily_user_traffic = [len(i) for i in result_dfs[day_idx:]]
            sum_user_traffic = sum(daily_user_traffic)


            daily_rho = []

            for p in range(dataset.provider_num):
                k = 1.1 * (total_estate[p] / sum_user_traffic) #2
                claims = [k * i for i in daily_user_traffic]
                daily_rho.append(bkr.allocate(total_estate[p], claims))

            daily_rho = np.array(daily_rho)


        hour_user_traffic = len(subset_df)
        batch_UI = sig_train_ui_matrix[subset_df[dataset.uid_field]].reshape(hour_user_traffic, -1)  # use trianed ui matrix
        batch_UI = torch.Tensor(batch_UI).to('cuda')
        if args.fairness_model == 'Bankruptcy':
            recommend_items, beta = model.recommendation(batch_UI, test_providerLen, daily_rho=daily_rho[:, 0])

        if isinstance(recommend_items, torch.Tensor):
            recommend_items = recommend_items.tolist()


        for user, inter_time, topk_items in tqdm(zip(subset_df[dataset.uid_field], subset_df.index,
                                                recommend_items), desc="Evaluating", total=len(recommend_items)):
            if not isinstance(topk_items, list):
                topk_items = topk_items.tolist()

            one_UI = sig_train_ui_matrix[user, :]
            ndcg_sui = NDCG(one_UI, topk_items)

            new_dict = {'userid:token': user,
                        'interaction_time': inter_time,
                        'NDCG_sui':ndcg_sui,
                        'recommend_list': topk_items}
            new_df = pd.DataFrame([new_dict])
            result_df = pd.concat([result_df, new_df])  # result_df.append(new_row, ignore_index=True)
            # print(result_df)
            result_df.to_csv(df_filename, index=False, mode='a', header=False)
            result_df = pd.DataFrame(columns=column_names)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_baseline")
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--lbd', type=float, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--start_date', type=str, default='2000-05-01')
    parser.add_argument('--end_date', type=str, default='2003-01-01')
    parser.add_argument('--window_size', type=int, default=30)  # yelp_small 60
    parser.add_argument('--file_path', type=str, default='../data')
    parser.add_argument('--fairness_model', type=str, default='P-MMF')
    parser.add_argument('--fairness_type', type=str, default='OF', help='choose among [UF, OF]')
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--base_model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--interval_len', type=str, default='24', help='time interval length')
    parser.add_argument('--total_estate', type=float, default=1000, help='total estate')
    parser.add_argument('--theta', type=float, default=0.25, help='theta of bank')
    parser.add_argument('--para', type=float, default=0.2, help='fairrec')
    args = parser.parse_args()
    args.config_files = os.path.join('recbole', 'properties', 'bankruptcy', f'{args.base_model}_{args.dataset}_top{str(args.topk)}.yaml')
    if args.dataset == 'KuaiRand-1K':
        args.start_date = '2022-04-22'
        args.end_date = '2022-05-08'

    dataset = DatasetStats()
    dataset.load_from_file(args.file_path)
    model = choose_model(dataset)
    run_baseline(dataset, model)






