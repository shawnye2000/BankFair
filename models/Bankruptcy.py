import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
import yaml
import torch
from sklearn.preprocessing import MinMaxScaler
import bankrupt
from datetime import datetime
from tqdm import tqdm,trange
import torch.nn.functional as F

class GPU_layer(nn.Module):
    def __init__(self,  p_size, c, rho):
        super(GPU_layer, self).__init__()
        self.rho = rho
        self.A = torch.triu(torch.ones((p_size,p_size)))
        self.d = torch.ones(p_size)
        self.c = c.cpu().numpy()
        self.p_size = p_size


    def forward(self,x):
        sorted_args = torch.argsort(x*self.rho.to(x.device),dim=-1)
        sorted_x = x.gather(dim=-1,index=sorted_args)

        rho = self.rho.gather(dim=-1,index=sorted_args).cpu()
        answer = cp.Variable(self.p_size)
        para_ordered_tilde_dual = cp.Parameter(self.p_size)
        constraints = []
        constraints += [answer + self.c >= 0]
        objective = cp.Minimize(cp.sum_squares(cp.multiply(rho,answer) - cp.multiply(rho, para_ordered_tilde_dual)))
        problem = cp.Problem(objective, constraints)
        #assert problem.is_dpp()
        self.cvxpylayer = CvxpyLayer(problem, parameters=[para_ordered_tilde_dual], variables=[answer])

        solution, = self.cvxpylayer(sorted_x)
        re_sort = torch.argsort(sorted_args,dim=-1)
        return solution.to(x.device).gather(dim=-1,index=re_sort)

def compute_next_dual(gradient, mu_t, provider_num, eta, c):
    if isinstance(gradient, torch.Tensor):
        gradient = gradient.cpu().numpy()
        mu_t = mu_t.cpu().numpy()
        c = c.cpu().numpy()
    answer = cp.Variable(provider_num)
    objective = cp.Minimize(cp.sum(cp.multiply(gradient, answer)) + eta * cp.sum_squares(answer - mu_t))
    constraints = [answer>=-c]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    #print(type(result))
    #exit(0)
    #print(type(answer.value))
    ans = answer.value
    ans = torch.FloatTensor(ans).to('cuda')
    return ans


def minmax_normalize(tensor, min_val=None, max_val=None):
    if min_val is None:
        min_val = torch.min(tensor)
    if max_val is None:
        max_val = torch.max(tensor)

    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out



class Bankruptcy(object):
    def __init__(self, rho, TopK, M, item_num, provider_item_counts, base_model,theta, fairness_model):
        self.TopK = TopK
        self.device = 'cuda'
        self.base_model = base_model
        self.fairness_model = fairness_model
        # self.A = M #[item_num, provider_num]
        self.A = torch.FloatTensor(M).to(self.device)
        self.A_sparse = torch.FloatTensor(M).to(self.device).to_sparse()
        _ , self.num_providers = self.A.shape
        f = open("properties/Bankruptcy.yaml", 'r')
        self.item_num = item_num
        self.provider_item_counts = torch.FloatTensor(provider_item_counts).to(self.device)
        self.uni_provider_item_counts = torch.FloatTensor(provider_item_counts/sum(provider_item_counts)).to(self.device)
        self.hyper_parameters = yaml.load(f)
        self.lambd = self.hyper_parameters['lambda']
        self.gamma = self.hyper_parameters['gamma']
        self.learning_rate = self.hyper_parameters['learning_rate']
        self.time_interval_len = 0.5
        self.time_interval_user_traffic = 0
        self.rho = torch.FloatTensor(rho).to(self.device)
        self.E_O = 0
        self.E_P = 0

        self.eta_t = 0.1
        self.time = datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.mu_t = np.zeros(self.num_providers)  #.to(self.device)
        self.delta_t = 0
        self.theta = theta
        print(f'theta:{self.theta}')
        # self.c_t = self.theta * F.softmax(1/self.uni_provider_item_counts)
        self.flag = True
        self.M = 1000
        self.acc_history = torch.zeros(self.num_providers).to(self.device)

    def data_prepare(self, dataset):
        if dataset == 'Amazon_Beauty':
            start_date = '2003-11-22'
            end_date = '2014-07-23'
        elif dataset == 'yelp':
            start_date = '2006-01-01'
            end_date = '2009-01-01'
        elif dataset == 'KuaiRec':
            start_date = '2020-07-05'
            end_date = '2020-08-25'
        elif dataset == 'small_KuaiRec':
            start_date = '2020-07-05'
            end_date = '2020-08-14'
        df = pd.read_table(os.path.join('../data', dataset, dataset + '.inter'))
        user_field, item_field, label_field, time_field, provider_field = df.columns
        df['interaction_time'] = pd.to_datetime(df[time_field], unit='s')
        daily_interactions = df.groupby(df['interaction_time'].dt.to_period(str(24*self.time_interval_len)+'H'))[user_field].count()
        daily_interactions = daily_interactions[start_date:end_date]
        daily_interactions = daily_interactions.dropna()
        daily_interactions = daily_interactions[::1]
        daily_interactions.index = daily_interactions.index.to_timestamp()
        return daily_interactions


    def normal_recommendation(self, batch_UI, t):
        batch_UI_matrix = batch_UI.reshape(1, -1)
        user_size, item_size = batch_UI_matrix.shape

        x = cp.Variable((user_size, item_size), integer=True)

        con = [x >= 0, x <= 1]

        for u in range(user_size):
            con.append(cp.sum(x[u, :]) == self.Topk)

        user_provider_counts = cp.matmul(x, self.A)
        e = cp.sum(user_provider_counts, axis=0)
        c = np.ones(e.shape)
        regularizer = -1 * cp.sum(cp.multiply(c, cp.maximum(t - e, 0)))
        if self.lambd == 0:
            obj = cp.sum(cp.multiply(batch_UI_matrix, x))
        else:
            obj = cp.sum(cp.multiply(batch_UI_matrix, x)) + self.lambd * regularizer

        obj = cp.Maximize(obj)

        prob = cp.Problem(obj, con)
        prob.solve(solver='MOSEK')
        print(f'obj:', prob.value)
        # print('最优解：', x.value)
        return np.argwhere(x.value == 1)[:, 1]


    def recommendation(self, batch_UI, test_providerLen, daily_rho):
        daily_rho = torch.FloatTensor(daily_rho).to(self.device)
        # provider_demand = [daily_rho for p in range(self.num_providers)]
        history = daily_rho

        if self.device =='cuda':
            batch_size = len(batch_UI)  # batch size means the predicted user size

            provider_num = self.A.shape[1]
            eta = self.learning_rate / math.sqrt(self.item_num)

            mu_t = torch.zeros(provider_num).to(self.device)
            e_t = torch.zeros(provider_num).to(self.device)
            # if torch.all(self.acc_beta < 0):
            #     self.c_t =  self.theta  *0.3 * F.softmax(torch.zeros(self.num_providers).to(self.device), dim=0)
            # else:
            #
            # if self.flag == True:
            #     self.c_t = self.theta * F.softmax(self.acc_beta, dim=0)
            #     self.flag = False
            # else:
            #     pos_acc_beta = torch.clamp(self.acc_beta, min=0)
            # self.c_t = self.theta *(0.5 * daily_rho/ 1000 + 0.5 * F.softmax(torch.zeros(self.num_providers).to(self.device), dim=0) )   #pos_acc_beta/torch.sum(pos_acc_beta)#F.softmax(self.acc_beta, dim=0)   #     #F.softmax(self.acc_beta, dim=0)  # torch.full((provider_num,), 100).to(self.device) #torch.ones(provider_num).to(self.device)
            # if self.fairness_model == 'Bankruptcy' or self.fairness_model == 'Naive':
            #     beta = 0.4
            # else:
            #     beta = 0.6
            beta = 0.4  # 0.4
            self.c_t = (beta * (1 / self.provider_item_counts)/torch.max(1/self.provider_item_counts) +
                        (1-beta) *torch.FloatTensor([2 / self.num_providers for i in range(self.num_providers)]).to(self.device))
            self.c_t = self.theta * self.c_t

            # print(f'c_t:penalty{self.c_t}')
            t_t = daily_rho
            # print(f'daily  rho:{daily_rho}')
            # rho_t = t_t * 100000
            # self.rho = rho_t
            # self.update_mu_func = GPU_layer(p_size=self.num_providers, c=self.c_t, rho=self.rho)
            B_t = batch_size * self.TopK * self.rho  #
            rho_t = B_t
            gradient_cusum = torch.zeros(provider_num).to(self.device)
            recommended_list = []
            for t in trange(batch_size):
                x_title = batch_UI[t, :] - self.A_sparse.matmul(mu_t.t()).t()
                mask = self.A_sparse.matmul((B_t > 0).float().t()).t()

                mask = (1.0 - mask) * -10000.0
                values, items = torch.topk(x_title+mask, k=self.TopK, dim=-1)

                re_allocation = torch.argsort(batch_UI[t, items], descending=True)
                x_allocation = items[re_allocation]
                recommended_list.append(x_allocation)

                # choose the optimal action
                for p in range(provider_num):
                    if mu_t[p] >= - self.c_t[p] and mu_t[p] < 0:
                        e_t[p] = t_t[p]
                    elif mu_t[p] >= 0:
                        e_t[p] = rho_t[p]

                # updating the remaining resource
                B_t = B_t - torch.sum(self.A[x_allocation], dim=0, keepdims=False)
                history = history - torch.sum(self.A[x_allocation], dim=0, keepdims=False)
                gradient_tidle = -torch.mean(self.A[x_allocation], dim=0, keepdims=False) + e_t

                gradient = self.gamma * gradient_tidle + (1 - self.gamma) * gradient_cusum
                gradient_cusum = gradient

                for g in range(1):
                    mu_t = compute_next_dual(gradient, mu_t, provider_num, eta, self.c_t) # self.update_mu_func(mu_t - eta * gradient / self.rho / self.rho) #
            history = torch.clamp(history, min=0)
            self.acc_history += history #/ daily_rho
            return recommended_list, history
        else:

            batch_UI = batch_UI.cpu().numpy()
            batch_size = len(batch_UI)  # batch size means the predicted user size

            provider_num = self.A.shape[1]
            eta = self.learning_rate / math.sqrt(self.item_num)

            mu_t = np.zeros(provider_num)
            e_t = np.zeros(provider_num)
            c_t = np.ones(provider_num)
            t_t = np.array(daily_rho)
            rho_t = t_t * 2
            self.rho = t_t * 2
            B_t = batch_size * self.TopK * self.rho
            gradient_cusum = np.zeros(provider_num)
            recommended_list = []
            for t in range(batch_size):
                x_title = batch_UI[t,:] - np.matmul(self.A, mu_t) #x_title.shape=[1,item_num]  mu_t.shape=[provider_num,1]
                mask = np.matmul(self.A, (B_t > 0).astype(np.float))
                mask = (1.0 - mask) * -10000.0
                x = np.argsort(x_title+mask, axis=-1)[::-1]
                x_allocation = x[:self.TopK]
                re_allocation = np.argsort(batch_UI[t, x_allocation])[::-1]
                x_allocation = x_allocation[re_allocation]
                recommended_list.append(x_allocation)



                # choose the optimal action
                for p in range(provider_num):
                    if mu_t[p] >= - c_t[p] and mu_t[p] < 0:
                        e_t[p] = t_t[p]
                    elif mu_t[p] >=0:
                        e_t[p] = rho_t[p]

                # updating the remaining resource
                B_t = B_t - np.sum(self.A[x_allocation], axis=0, keepdims=False)


                gradient = -np.mean(self.A[x_allocation], axis=0, keepdims=False) + e_t
                gradient = self.gamma * gradient + (1 - self.gamma) * gradient_cusum
                gradient_cusum = gradient
                # gradient = -(B_0-B_t)/((t+1)*K) + rho
                for g in range(1):
                    mu_t = compute_next_dual(gradient, mu_t, provider_num, eta, c_t)
            return recommended_list




    def predict_user_traffic(self, new_data):
        input_size = 1  # n_features
        hidden_size = 24
        output_size = 7 / self.time_interval_len
        n_step = 50

        if self.flag == True:
            self.observe_window_data = self.data_prepare('small_KuaiRec')
            self.flag = False
        else:

            new_df = pd.DataFrame({'interaction_time': [new_data]})

            self.observe_window_data = pd.concat([self.observe_window_data, new_df])
        data_np = np.array(self.observe_window_data.values).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_scaled = scaler.fit_transform(data_np)

        X_data = data_scaled[-n_step:]
        X_data = torch.tensor(X_data, dtype=torch.float32).view(1, -1, 1)

        loaded_model = GRUModel(input_size, hidden_size, output_size)
        loaded_model.load_state_dict(torch.load('gru_model.pth'))
        loaded_model.eval()

        predict_data = loaded_model(X_data)

        return scaler.inverse_transform(predict_data.detach().numpy())


    def recommend_one_1(self, one_UI, time):
        # one_UI = torch.FloatTensor(one_UI).to(self.device)
        one_UI = one_UI.numpy()
        # if a new hour ?
        if time.hour > self.time.hour:

            self.E_P = self.predict_user_traffic(self.E_O)
            self.E_P = np.nan_to_num(self.E_P)
            k = 1
            claim = (k * self.provider_item_counts).tolist()
            # print(self.E_P, self.E_O)
            self.rho_t = np.array(bankrupt.allocate(self.E_P + self.E_O, claim))
            self.E_O = 0
            self.time = time
        else:
            self.E_O += 1


        # lagrange function

        x_t = cp.Variable(self.item_num, boolean=True)
        violation = self.rho_t - cp.matmul(self.A.T, x_t)
        obj = (cp.sum(cp.multiply(one_UI, x_t)) +
                     cp.sum(cp.multiply(self.mu_t, violation)) - np.linalg.norm(self.mu_t)/2)
        objective = cp.Minimize(obj)
        constraints = [cp.sum(x_t) == self.TopK]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        x_allocation = x_t.value

        recommended_list = np.argsort(-x_allocation*one_UI)[:self.TopK]

        # update mu_t
        exposure_allocated = np.matmul(self.A.T, x_allocation)
        self.rho_t = np.maximum(self.rho_t - exposure_allocated, 0)
        gradient = self.rho_t - exposure_allocated - self.delta_t * self.mu_t
        self.mu_t = np.maximum(self.mu_t + self.eta_t * gradient, 0)

        return recommended_list


    def recommend_one_cpu(self, one_UI, time, daily_user_num):
        # oracle
        # one_UI = torch.FloatTensor(one_UI).to(self.device)
        one_UI = one_UI.numpy()
        # if a new hour ?
        # if time.hour > self.time.hour:
        #
        #     self.E_P = self.predict_user_traffic(self.E_O)
        #     self.E_P = np.nan_to_num(self.E_P)
        #     k = 1
        #     claim = (k * self.provider_item_counts).tolist()
        #     # print(self.E_P, self.E_O)
        #     self.rho_t = np.array(bankrupt.allocate(self.E_P + self.E_O, claim))
        #     self.E_O = 0
        #     self.time = time
        # else:
        #     self.E_O += 1

        if time.day > self.time.day:
            self.mu_t = np.zeros(self.num_providers)
            self.rho_t = np.array([self.M / daily_user_num for i in range(self.num_providers)])

        # lagrange function

        x_t = cp.Variable(self.item_num, boolean=True)
        violation = self.rho_t - cp.matmul(self.A.T, x_t)
        obj = (cp.sum(cp.multiply(one_UI, x_t)) +
                     cp.sum(cp.multiply(self.mu_t, violation)) - np.linalg.norm(self.mu_t)/2)
        objective = cp.Minimize(obj)
        constraints = [cp.sum(x_t) == self.TopK]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        x_allocation = x_t.value

        recommended_list = np.argsort(-x_allocation*one_UI)[:self.TopK]

        # update mu_t
        exposure_allocated = np.matmul(self.A.T, x_allocation)
        # self.rho_t = np.maximum(self.rho_t - exposure_allocated, 0)
        gradient = self.rho_t - exposure_allocated - self.delta_t * self.mu_t
        self.mu_t = np.maximum(self.mu_t + self.eta_t * gradient, 0)

        return recommended_list



    def recommend_one(self, one_UI, time, daily_user_num):
        # oracle
        # one_UI = torch.FloatTensor(one_UI).to(self.device)
        one_UI = torch.FloatTensor(one_UI).to(self.device)

        # if a new hour ?
        # if time.hour > self.time.hour:
        #
        #     self.E_P = self.predict_user_traffic(self.E_O)
        #     self.E_P = np.nan_to_num(self.E_P)
        #     k = 1
        #     claim = (k * self.provider_item_counts).tolist()
        #     # print(self.E_P, self.E_O)
        #     self.rho_t = np.array(bankrupt.allocate(self.E_P + self.E_O, claim))
        #     self.E_O = 0
        #     self.time = time
        # else:
        #     self.E_O += 1

        if time.day > self.time.day:
            # print("new day!!!")
            E = self.M  #each provider's claim for the 7 days
            # d = k * user traffic for the day
            # rho_t = banrupt.allocate(E, d)
            d = self.predict_user_traffic(new_data =  self.time_interval_user_traffic)  # new data is the user traffic in the last time interval
            self.rho_t = torch.tensor(bankrupt.reverse_talmud_allocate(E,d)).to(self.device)
            self.mu_t = torch.zeros(self.num_providers).to(self.device)
            # self.rho_t = torch.tensor([self.M / daily_user_num for i in range(self.num_providers)]).to(self.device)
            self.time = time
        # lagrange function
        self.time_interval_user_traffic += 1
        # x_t = cp.Variable(self.item_num, boolean=True)
        # violation = self.rho_t - cp.matmul(self.A.T, x_t)
        # obj = (cp.sum(cp.multiply(one_UI, x_t)) +
        #              cp.sum(cp.multiply(self.mu_t.cpu(), violation)) - np.linalg.norm(self.mu_t.cpu())/2)
        # objective = cp.Minimize(obj)
        # constraints = [cp.sum(x_t) == self.TopK]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # x_allocation = x_t.value

        x_title = one_UI[:] + self.A.matmul(self.mu_t.t()).t()
        # x = np.argsort(x_title+mask,axis=-1)[::-1]
        values, items = torch.topk(x_title, k=self.TopK, dim=-1)
        # x_allocation = x[:self.TopK]
        re_allocation = torch.argsort(one_UI[items], descending=True)
        x_allocation = items[re_allocation]

        recommended_list = x_allocation

        # update mu_t
        exposure_allocated = torch.sum(self.A[x_allocation], dim=0, keepdims=False)
        # print(f'exposure allocated:{exposure_allocated}')
        # self.rho_t = np.maximum(self.rho_t - exposure_allocated, 0)
        gradient = self.rho_t - exposure_allocated - self.delta_t * self.mu_t
        self.mu_t = torch.clamp(self.mu_t + self.eta_t * gradient, min=0)
        # print(self.mu_t)
        return recommended_list
