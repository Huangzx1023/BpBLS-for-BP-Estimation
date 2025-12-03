import sys
import os
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from utils import show_regmetric, ridge_regression_pseudo_inverse_reg, normalize_data, apply_normalization, compute_mean_values
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Train
bp_dict = np.load('/path/to/project/train/tr_refbp.npy', allow_pickle=True).item()
train_dict = np.load('/path/to/project/train/tr_demo_sigfeas.npy', allow_pickle=True).item()
cal_dict = np.load('/path/to/project/train/tr_cal.npy', allow_pickle=True).item()
pp_dict = np.load('/path/to/project/train/tr_pp.npy', allow_pickle=True).item()

## Test
test_x = np.load('/path/to/project/test/ts_demo_sigfeas.npy', allow_pickle=True)
test_y = np.load('/path/to/project/test/ts_refbp.npy', allow_pickle=True)
test_cal = np.load('/path/to/project/test/ts_cal.npy')
test_index = np.load('/path/to/project/test/ts_index.npy')
test_index = torch.tensor(test_index, dtype=torch.float32)

# Convert data to PyTorch tensors
train_x = []
train_y = []
train_cal = []
train_pp = []
for j in range(5):
    train_x.append(train_dict[j])
    train_y.append(bp_dict[j])
    train_cal.append(cal_dict[j])
    train_pp.append(pp_dict[j])

train_x = np.concatenate(train_x)
train_y = np.concatenate(train_y)
train_cal = np.concatenate(train_cal)
train_pp = np.concatenate(train_pp)

train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
train_cal = torch.tensor(train_cal, dtype=torch.float32).to(device)
train_pp = torch.tensor(train_pp, dtype=torch.float32).to(device)

train_x = torch.cat((train_x, train_cal), dim=1)
test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
test_cal = torch.tensor(test_cal, dtype=torch.float32).to(device)
test_x = torch.cat((test_x, test_cal), dim=1)

def pinv(A, reg):
    identity_matrix = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    regularized_matrix = reg * identity_matrix + torch.matmul(A.T, A)
    regularized_inverse = torch.linalg.pinv(regularized_matrix)
    return torch.matmul(regularized_inverse, A.T)

class BLS(nn.Module):
    def __init__(self, feature_nodes, feature_windows, enhancement_nodes, num_classes, normalization='batch'):
        super(BLS, self).__init__()
        self.feature_nodes = feature_nodes
        self.feature_windows = feature_windows
        self.bn_in = nn.BatchNorm1d(74, track_running_stats=True)
        self.feature_layer = nn.Linear(74, feature_nodes * feature_windows)
        if normalization == 'batch':
            self.feature_bn = nn.BatchNorm1d(feature_nodes * feature_windows, track_running_stats=True)
        else:
            self.feature_bn = nn.Identity()
            
        self.fc31 = nn.Linear(feature_nodes * feature_windows, enhancement_nodes)
        self.bn_en = nn.BatchNorm1d(enhancement_nodes, track_running_stats=True)
        self.final = nn.Linear(feature_nodes * feature_windows + enhancement_nodes, num_classes, bias=False)

    def forward(self, x, y):
        x = self.bn_in(x)
        feature_nodes = self.feature_layer(x)
        feature_nodes = self.feature_bn(feature_nodes)
        
        feature_nodes = torch.sigmoid(feature_nodes)

        enhancement_nodes = self.fc31(feature_nodes)
        enhancement_nodes = self.bn_en(enhancement_nodes)
        enhancement_nodes = torch.tanh(enhancement_nodes)

        FeaAndEnhance = torch.cat([feature_nodes, enhancement_nodes], dim=1)
        outs = self.final(FeaAndEnhance)
        return outs, FeaAndEnhance


class BLSincre:
    def __init__(self, traindata, trainlabel, extratraindata, extratrainlabel, model, alpha1,
                 pesuedoinverse, W, traincal, extratraincal, beta, pp, extra_pp):  # 都得是tensor
        self.traindata = traindata
        self.trainlabel = trainlabel
        self.extratraindata = extratraindata
        self.extratrainlabel = extratrainlabel
        self.model = model
        self.W = W.T
        self.pesuedoinverse = pesuedoinverse
        self.c = alpha1
        self.traincal = traincal
        self.extratraincal = extratraincal
        self.beta = beta
        self.pp = pp
        self.extra_pp = extra_pp


        self.incremental_input()

    def incremental_input(self):
        self.W = self.W.T
        self.pesuedoinverse = self.pesuedoinverse * (torch.tensor(self.beta) + 1)
        ## extratraindata
        self.model.eval()
        _, data = self.model(self.traindata, self.traincal)
        _, xdata = self.model(self.extratraindata.to(torch.float32), self.extratraincal)
        xdata = xdata.T
        xlabel = self.extratrainlabel.T.to(torch.float32)
        # Update
        DT = xdata.T @ self.pesuedoinverse
        CT = xdata.T - DT @ data

        if (CT.T == 0).all():
            B = pinv(CT, self.c)
        else:
            DT_DT_T = torch.matmul(DT, DT.T)
            identity_matrix = torch.eye(DT.shape[0], device=DT.device, dtype=DT.dtype)
            inv_term = torch.linalg.pinv(DT_DT_T + identity_matrix)
            B = torch.matmul(self.pesuedoinverse, torch.matmul(DT.T, inv_term))

        self.pesuedoinverse = torch.cat((self.pesuedoinverse - torch.matmul(B, DT), B), dim=1)/(torch.tensor(self.beta)+1)

        # Define
        Beta = self.beta / (1 + self.beta)
        Beta2 = (1 + self.beta) / (1 + 2*self.beta)
        W_d_old = self.W[:, 0:1]
        W_s_old = self.W[:, 1:2]
        C = B
        A_new = xdata
        new_label = xlabel.T
        y_d_new = Beta2 * (new_label[:, 0:1] + Beta * new_label[:, 1:2] - Beta * self.extra_pp)
        y_s_new = Beta2 * (Beta * new_label[:, 0:1] + new_label[:, 1:2] + Beta * self.extra_pp)
        W_d = W_d_old + C @ (y_d_new - A_new.T @ W_d_old)
        W_s = W_s_old + C @ (y_s_new - A_new.T @ W_s_old)
        self.W = torch.cat((W_d, W_s), dim=1)

    def incremental_predict(self, test_x, test_y):
        return self.W, self.pesuedoinverse

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# store all model performance metrics
performance_results = []

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
EPOCHS = 50
seed_everything(seed=42)
net = BLS(feature_nodes=10, feature_windows=46, enhancement_nodes=28, num_classes=2)
print(net.to(device))

lossF = torch.nn.MSELoss()
alpha = 2 ** -8
beta = 2 ** -7
optimizer = torch.optim.Adam(net.parameters(), lr=10**0, weight_decay=2**-12)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.90)
i = 0
for epoch in tqdm(range(0, EPOCHS + 1), position=0, file=sys.stdout, desc="Training progress"):
    if epoch < EPOCHS:
        net.train()
        outputs, _ = net(train_x, train_cal)
        outputs = outputs * 0.29 + train_cal * (1 - 0.29)
        loss1 = lossF(outputs, train_y)
        loss2 = lossF(outputs[:, 1:]-outputs[:, 0:1], train_pp)
        loss = loss1 + beta*loss2
        net.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        mse_low, rmse_low, nrmse_low, mae_low, me_low, mesd_low, mse_high, rmse_high, nrmse_high, mae_high, me_high, mesd_high = show_regmetric(
            outputs, train_y)

        net.eval()
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        outputs, _ = net(test_x, test_cal)
        _, outputs = compute_mean_values(outputs, test_index)
        outputs = outputs * 0.29 + test_cal * (1 - 0.29)
        mse_low, rmse_low, nrmse_low, mae_low, me_low, mesd_low, mse_high, rmse_high, nrmse_high, mae_high, me_high, mesd_high = show_regmetric(
            outputs, test_y)

        know_x = train_x.to(device)
        know_y = train_y.to(device)
        know_cal = train_cal.to(device)

    ############################ Final epoch #############################################
    #### read data
    if epoch == EPOCHS:
        print("\nFinal epoch:\n")
        net.eval()
        with torch.no_grad():
            _, FeaAndEnhance = net(know_x, know_cal)
        pinv_re, new_params = ridge_regression_pseudo_inverse_reg(FeaAndEnhance, know_y, alpha, beta, train_pp)
        W = new_params
        net.eval()
        _, outputs = net(test_x, test_cal)
        outputs = torch.matmul(outputs, W)
        _, outputs = compute_mean_values(outputs, test_index)
        outputs = outputs* 0.29 + test_cal*(1- 0.29)
        mse_low, rmse_low, nrmse_low, mae_low, me_low, mesd_low, mse_high, rmse_high, nrmse_high, mae_high, me_high, mesd_high = show_regmetric(
            outputs, test_y)
        print("------------------- Before increment -------------------".format(i + 1))
        print(" -- LOW")
        print("MSE: {:.4f}".format(mse_low),
              "RMSE: {:.4f}".format(rmse_low),
              "nRMSE: {:.4f}".format(nrmse_low),
              "MAE: {:.4f}".format(mae_low),
              "ME: {:.4f}".format(me_low),
              "ME_sd: {:.4f}".format(mesd_low), )
        print(" -- HIGH")
        print("MSE: {:.4f}".format(mse_high),
              "RMSE: {:.4f}".format(rmse_high),
              "nRMSE: {:.4f}".format(nrmse_high),
              "MAE: {:.4f}".format(mae_high),
              "ME: {:.4f}".format(me_high),
              "ME_sd: {:.4f}".format(mesd_high), )

        ###### incremental
        for i in range(3):
            i += 1
            temp_x = []
            temp_y = []
            temp_cal = []
            temp_pp = []
            if i == 1:
                j_range = range(5, 12)
            elif i == 2:
                j_range = range(12, 22)
            elif i == 3:
                j_range = range(22, 54)

            for j in j_range:
                temp_x.append(train_dict[j])
                temp_y.append(bp_dict[j])
                temp_cal.append(cal_dict[j])
                temp_pp.append(pp_dict[j])

            temp_x = np.concatenate(temp_x)
            temp_y = np.concatenate(temp_y)
            temp_cal = np.concatenate(temp_cal)
            temp_pp = np.concatenate(temp_pp)
            temp_x = torch.tensor(temp_x, dtype=torch.float32).to(device)
            temp_y = torch.tensor(temp_y, dtype=torch.float32).to(device)
            temp_cal = torch.tensor(temp_cal, dtype=torch.float32).to(device)
            temp_pp = torch.tensor(temp_pp, dtype=torch.float32).to(device)
            temp_x = apply_normalization(temp_x, data_min, data_max)
            temp_x = torch.cat((temp_x, temp_cal), dim=1)

            net.eval()
            train_pp = torch.cat((train_pp, temp_pp), dim=0)
            BLSincre1 = BLSincre(traindata=know_x, trainlabel=know_y, extratraindata=temp_x, extratrainlabel=temp_y,
                                model=net, alpha1=alpha, pesuedoinverse=pinv_re, W=W, traincal = know_cal, extratraincal = temp_cal, beta=beta, pp=train_pp, extra_pp=temp_pp)
            W, pinv_re = BLSincre1.incremental_predict(test_x, test_y)
            know_x = torch.cat((know_x, temp_x), dim=0)
            know_y = torch.cat((know_y, temp_y), dim=0)
            know_cal = torch.cat((know_cal, temp_cal), dim=0)

            #### Test
            net.eval()
            _, outputs = net(test_x, test_cal)
            outputs = torch.matmul(outputs, W)
            _, outputs = compute_mean_values(outputs, test_index)
            outputs = outputs* 0.3 + test_cal*(1- 0.3)
            mse_low, rmse_low, nrmse_low, mae_low, me_low, mesd_low, mse_high, rmse_high, nrmse_high, mae_high, me_high, mesd_high = show_regmetric(
                outputs, test_y)
            print("------------------- Increment {:d} -------------------".format(i))
            print(" -- LOW")
            print("MSE: {:.4f}".format(mse_low),
                  "RMSE: {:.4f}".format(rmse_low),
                  "nRMSE: {:.4f}".format(nrmse_low),
                  "MAE: {:.4f}".format(mae_low),
                  "ME: {:.4f}".format(me_low),
                  "ME_sd: {:.4f}".format(mesd_low), )
            print(" -- HIGH")
            print("MSE: {:.4f}".format(mse_high),
                  "RMSE: {:.4f}".format(rmse_high),
                  "nRMSE: {:.4f}".format(nrmse_high),
                  "MAE: {:.4f}".format(mae_high),
                  "ME: {:.4f}".format(me_high),
                  "ME_sd: {:.4f}".format(mesd_high), )



