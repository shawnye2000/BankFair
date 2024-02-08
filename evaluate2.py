import torch
import argparse
user_field, time_field, acc_field, ndcg_sui_field, ndcg_field, recall_field, hr_field, \
        ndcg2_field, recall2_field, hr2_field, list_field = ('userid:token', 'interaction_time','Accuracy','NDCG_sui',
                    'NDCG', 'recall', 'hitrate',
                    'NDCG2', 'recall2', 'hitrate2',
                   'recommend_list')
import pandas as pd
from torch import tensor
import json
import os
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DatasetStats:
    def __init__(self):
        self.user_num = 0
        self.item_num = 0
        self.provider_num = 0
        self.UI_matrix = None
        self.A = None
        self.item2provider = {}
        self.provider_item_counts = None
        self.provideLen = None # provider interaction times
        self.inter_dataframe = None
        self.uid_field = ''
        self.iid_field = ''
        self.label_field = ''
        self.time_field = ''
        self.provider_field = ''

    def load_from_file(self, file_path, dataset):
        processed_data_path = os.path.join(file_path, dataset, dataset + '.inter')
        self.inter_dataframe = pd.read_csv(processed_data_path, sep='\t')
        # table head:review_id:token	user_id:token	business_id:token	stars:float	useful:float funny:float	cool:float	date:float
        print(f'interaction times:{len(self.inter_dataframe)}')
        self.uid_field, self.iid_field, self.label_field, self.time_field, self.provider_field = self.inter_dataframe.columns
        self.inter_dataframe['interaction_time'] = pd.to_datetime(self.inter_dataframe[self.time_field], unit='ms')
        self.inter_dataframe = self.inter_dataframe.sort_values(by='interaction_time', ascending=True) # 递增
        target_date = pd.to_datetime('2022-04-22')
        after_df = self.inter_dataframe[self.inter_dataframe['interaction_time']>=target_date]
        print(f'interaction times after:{len(after_df)}')
        self.provider_num = len(self.inter_dataframe[self.provider_field].unique())
        self.user_num, self.item_num = len(self.inter_dataframe[self.uid_field].unique()),len(self.inter_dataframe[self.iid_field].unique())
        print(f'user_num {self.user_num} item_num {self.item_num}')
        self.providerLen = np.array(self.inter_dataframe.groupby(self.provider_field).size().values)
        print(f'providerLen:{self.providerLen}')

        tmp = self.inter_dataframe[[self.iid_field, self.provider_field]].drop_duplicates()
        unique_item_datas = self.inter_dataframe.drop_duplicates(subset=[self.iid_field])
        self.provider_item_counts = np.array(unique_item_datas.groupby(self.provider_field)[self.iid_field].count())
        print(f'provider-item-count:{self.provider_item_counts}')
        self.item2provider = {x: y for x, y in zip(tmp[self.iid_field], tmp[self.provider_field])}
        A = np.zeros((self.item_num, self.provider_num))
        iid2pid = []
        for i in range(self.item_num):

            iid2pid.append(self.item2provider[i])
            A[i, self.item2provider[i]] = 1
        self.A = np.array(A)
        print(f'item-provider adjacency matrix{self.A.shape}')




# 求 ESG
def ESG(provider_exposure):
    # print(f'provider exposure{provider_exposure}')

    print(provider_exposure.values)
    print(f'min provider exp:{min(provider_exposure)}')
    satisfied_pro = 0
    provider_sum = len(provider_exposure)
    for i in provider_exposure:
        if i >= 1000:
            satisfied_pro += 1
    unsatisfied_providers = []
    for i,z in enumerate(provider_exposure):
        if z < 1000:
            unsatisfied_providers.append(i)
    print(unsatisfied_providers)
    return satisfied_pro / provider_sum


def provider_esg(df, dataset):
    import numpy as np
    inter_times = len(df)
    provider_exp = np.zeros((inter_times, dataset.provider_num))
    # df[list_field] = df[list_field].apply(lambda x : torch.tensor(eval(x)).to_list())

    for x, (idex, rec_list) in enumerate(df[list_field].items()):
        # rec_list = rec_list.strip('[]')
        # rec_list = rec_list.split(',')
        # print(rec_list)
        try:
            rec_list = eval(rec_list)
            if isinstance(rec_list, torch.Tensor):
                rec_list = tensor(rec_list).tolist()
            if isinstance(rec_list, list):
                rec_list = rec_list
        except:
            rec_list = rec_list.strip('[]').split()

        for item in rec_list:
            item = int(item)
            # try:

            provider_exp[x, dataset.item2provider[item]] += 1
            # except:
            #     continue

    df_provider = pd.DataFrame(provider_exp, columns=[f'provider_{i}' for i in range(provider_exp.shape[1])])

    hourly_esg = ESG(df_provider.sum())
    return hourly_esg


if __name__== '__main__':
    # parser = argparse.ArgumentParser(description="run_baseline")
    # parser.add_argument('--fairness_model', type=str, default='P-MMF')
    # parser.add_argument('--total_estate', type=float, default=1000, help='total estate')
    # args = parser.parse_args()
    interval_len = '24'
    fairness_model = 'P-MMF'
    dataset_name = 'KuaiRand-1K'
    lbd = '100.0'
    total_estate = '1000.0'
    base_model = 'BPR'
    theta = '0.15'
    para = '2.5'
    topk = '5'
    dataset = DatasetStats()
    dataset.load_from_file('../../data', dataset_name)
    if fairness_model == 'Bankruptcy' or fairness_model =='Prop' or fairness_model == 'Naive'or fairness_model == 'Consis':
        df_filename = '../df/run_baseline4' + fairness_model + '_'+dataset_name+ '_'+base_model + f'_interval{interval_len}_top{topk}_estate{total_estate}_theta{theta}'+'.csv'
    elif fairness_model == 'P-MMF' or fairness_model == 'CPFair' or fairness_model == 'pct' or \
         fairness_model == 'reg_exp' or fairness_model == 'Welf':
        df_filename = '../df/run_baseline4' + fairness_model + '_' + dataset_name + '_' + base_model + f'_interval{interval_len}_top{topk}_lambda{lbd}' + '.csv'
    elif fairness_model == 'FairRec':
        df_filename = '../df/run_baseline4' + fairness_model + '_' + dataset_name + '_' + base_model + f'_interval{interval_len}_top{topk}_para{para}' + '.csv'
    else:
        df_filename = '../df/run_baseline4' + fairness_model + '_' + dataset_name + '_' + base_model + f'_interval{interval_len}_top{topk}' + '.csv'
    df = pd.read_csv(df_filename, sep=',')
    df[time_field] = pd.to_datetime(df[time_field])
    # interval_result = df.groupby(df[time_field].dt.to_period(interval_len + 'H'))[[ndcg_sui_field]].mean().reset_index()  #  acc_field, ndcg2_field, recall2_field, hr2_field
    df.set_index('interaction_time', inplace=True)
    time_interval_len = interval_len + 'H'

    interval_result = df.resample(time_interval_len)[[ndcg_sui_field]].mean()
    expected_ndcg  = interval_result[ndcg_sui_field].mean()
    print(f'daily ndcg:{interval_result[ndcg_sui_field]}')
    # import matplotlib.pyplot as plt
    # plt.plot(interval_result[ndcg_sui_field])
    # plt.show()
    var_ndcg = interval_result[ndcg_sui_field][:].std()
    # expected_ndcg2  = interval_result[ndcg2_field].mean()
    # var_ndcg2 = interval_result[ndcg2_field].var()
    esg = provider_esg(df, dataset)
    print(f'e ndcg:{expected_ndcg}')
    print(f'var ndcg:{var_ndcg}')
    # print(f'e ndcg2:{expected_ndcg2}')
    # print(f'var ndcg2:{var_ndcg2}')
    print(f'esg :{esg}')

