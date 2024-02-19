import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
import torch.nn as nn
import yaml
import torch
from datetime import datetime
from tqdm import tqdm,trange


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
    ans = answer.value
    ans = torch.FloatTensor(ans).to('cuda')
    return ans



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
        self.beta = self.hyper_parameters['beta']
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
        self.flag = True
        self.M = 1000
        self.acc_history = torch.zeros(self.num_providers).to(self.device)



    def recommendation(self, batch_UI, test_providerLen, daily_rho):
        daily_rho = torch.FloatTensor(daily_rho).to(self.device)
        history = daily_rho

        batch_size = len(batch_UI)  # batch size means the predicted user size

        provider_num = self.A.shape[1]
        eta = self.learning_rate / math.sqrt(self.item_num)

        mu_t = torch.zeros(provider_num).to(self.device)
        e_t = torch.zeros(provider_num).to(self.device)

        self.c_t = (self.beta * (1 / self.provider_item_counts)/torch.max(1/self.provider_item_counts) +
                    (1-self.beta) *torch.FloatTensor([1 / self.num_providers for i in range(self.num_providers)]).to(self.device))
        self.c_t = self.theta * self.c_t

        t_t = daily_rho
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



