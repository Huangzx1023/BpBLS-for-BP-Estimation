import numpy as np
import torch
def show_regmetric(pred, true):
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    # pred = pred.numpy()
    # true = true.numpy()
    true = true.reshape(-1, 2)

    # Low pressure
    # MSE
    mse_low = np.mean((pred[:, 0] - true[:, 0]) ** 2)
    # RMSE
    rmse_low = np.sqrt(mse_low)
    # nRMSE
    nrmse_low = np.mean(np.sqrt((pred[:, 0] - true[:, 0]) ** 2) / np.sqrt(true[:, 0] ** 2))
    # MAE
    mae_low = np.mean(np.abs(pred[:, 0] - true[:, 0]))
    # ME
    me_low = np.mean(pred[:, 0] - true[:, 0])
    # ME_SD
    mesd_low = np.std(pred[:, 0] - true[:, 0])

    # High pressure
    # MSE
    mse_high = np.mean((pred[:, 1] - true[:, 1]) ** 2)
    # RMSE
    rmse_high = np.sqrt(mse_high)
    # nRMSE
    nrmse_high = np.mean(np.sqrt((pred[:, 1] - true[:, 1]) ** 2) / np.sqrt(true[:, 1] ** 2))
    # MAE
    mae_high = np.mean(np.abs(pred[:, 1] - true[:, 1]))
    # ME
    me_high = np.mean(pred[:, 1] - true[:, 1])
    # ME_SD
    mesd_high = np.std(pred[:, 1] - true[:, 1])
    return mse_low, rmse_low, nrmse_low, mae_low, me_low, mesd_low, mse_high, rmse_high, nrmse_high, mae_high, me_high, mesd_high

def ridge_regression_pseudo_inverse(X, y, alpha):
    """
    计算带有正则化的伪逆（岭回归）
    X: 输入特征矩阵
    y: 输出标签矩阵
    alpha: 正则化参数
    """
    n_features = X.shape[1]
    identity_matrix = torch.eye(n_features, device=X.device)
    XTX = torch.matmul(X.T, X)
    ridge_term = alpha * identity_matrix
    regularized_inverse = torch.linalg.pinv(XTX + ridge_term)
    pinv_re = torch.matmul(regularized_inverse, X.T)
    return pinv_re, torch.matmul(pinv_re, y)

def ridge_regression_pseudo_inverse_reg(X, y, alpha, beta, pp):
    """
    计算带有正则化的伪逆（岭回归）
    X: 输入特征矩阵
    y: 输出标签矩阵
    alpha: 正则化参数
    """
    n_features = X.shape[1]
    identity_matrix = torch.eye(n_features, device=X.device)
    XTX = torch.matmul(X.T, X)
    ridge_term = alpha * identity_matrix
    B_plus = torch.linalg.pinv((1 + beta) * XTX + ridge_term) @ X.T # regularized_inverse
    # low
    Beta = beta/(1+beta)
    C1 = 1.0/(1.0-Beta**2.0)
    delta = Beta*pp

    W_s = C1 * B_plus @ (Beta*y[:, 0:1] + y[:, 1:2] + delta)
    W_d = C1 * B_plus @ (y[:, 0:1] + Beta*y[:, 1:2] - delta)


    W = torch.cat((W_d, W_s), dim=1)

    return B_plus, W

def normalize_data(data):
    """
    归一化数据并返回归一化后的数据和最大最小值
    """
    data_min = torch.min(data, dim=0, keepdim=True)[0]
    data_max = torch.max(data, dim=0, keepdim=True)[0]
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data, data_min, data_max

def apply_normalization(data, data_min, data_max):
    """
    使用给定的最大最小值对数据进行归一化
    """
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

def show_regmetric_sk(pred, true):
    # Low pressure
    # MSE
    mse_low = np.mean((pred[:, 0] - true[:, 0]) ** 2)
    # RMSE
    rmse_low = np.sqrt(mse_low)
    # nRMSE
    nrmse_low = np.mean(np.sqrt((pred[:, 0] - true[:, 0]) ** 2) / np.sqrt(true[:, 0] ** 2))
    # MAE
    mae_low = np.mean(np.abs(pred[:, 0] - true[:, 0]))
    # ME
    me_low = np.mean(pred[:, 0] - true[:, 0])
    # ME_SD
    mesd_low = np.std(pred[:, 0] - true[:, 0])

    # High pressure
    # MSE
    mse_high = np.mean((pred[:, 1] - true[:, 1]) ** 2)
    # RMSE
    rmse_high = np.sqrt(mse_high)
    # nRMSE
    nrmse_high = np.mean(np.sqrt((pred[:, 1] - true[:, 1]) ** 2) / np.sqrt(true[:, 1] ** 2))
    # MAE
    mae_high = np.mean(np.abs(pred[:, 1] - true[:, 1]))
    # ME
    me_high = np.mean(pred[:, 1] - true[:, 1])
    # ME_SD
    mesd_high = np.std(pred[:, 1] - true[:, 1])
    return mse_low, rmse_low, nrmse_low, mae_low, me_low, mesd_low, mse_high, rmse_high, nrmse_high, mae_high, me_high, mesd_high

def normalize_data(data):
    """
    归一化数据并返回归一化后的数据和最大最小值
    """
    data_min = torch.min(data, dim=0, keepdim=True)[0]
    data_max = torch.max(data, dim=0, keepdim=True)[0]
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data, data_min, data_max

def normalize_data_sk(data):
    """
    归一化数据并返回归一化后的数据和最大最小值
    """
    data_min = np.min(data, axis=0, keepdims=True)[0]
    data_max = np.max(data, axis=0, keepdims=True)[0]
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data, data_min, data_max

def print_metrics(mse_low, rmse_low, nrmse_low, mae_low, me_low, mesd_low,
                  mse_high, rmse_high, nrmse_high, mae_high, me_high, mesd_high):
    print(" -- LOW")
    print("MSE: {:.4f}".format(mse_low),
          "RMSE: {:.4f}".format(rmse_low),
          "nRMSE: {:.4f}".format(nrmse_low),
          "MAE: {:.4f}".format(mae_low),
          "ME: {:.4f}".format(me_low),
          "ME_sd: {:.4f}".format(mesd_low))
    print(" -- HIGH")
    print("MSE: {:.4f}".format(mse_high),
          "RMSE: {:.4f}".format(rmse_high),
          "nRMSE: {:.4f}".format(nrmse_high),
          "MAE: {:.4f}".format(mae_high),
          "ME: {:.4f}".format(me_high),
          "ME_sd: {:.4f}".format(mesd_high))
    print(" ---------------------------------------------------------------------------------  \n")


def print_metrics_dl(mse_dbp, rmse_dbp, nrmse_dbp, mae_dbp, me_dbp, mesd_dbp,
                  mse_sbp, rmse_sbp, nrmse_sbp, mae_sbp, me_sbp, mesd_sbp):
    # 输出 SBP 部分
    print("SBP: MAE: {:.2f}, ME±SDE: {:.2f}±{:.2f}, RMSE: {:.2f}".format(mae_sbp, me_sbp, mesd_sbp, rmse_sbp))

    # 输出 DBP 部分
    print("DBP: MAE: {:.2f}, ME±SDE: {:.2f}±{:.2f}, RMSE: {:.2f}".format(mae_dbp, me_dbp, mesd_dbp, rmse_dbp))

    # 输出并列行，不带名称
    print("{:.2f}, {:.2f}±{:.2f}, {:.2f}, {:.2f}, {:.2f}±{:.2f}, {:.2f}".format(
        mae_sbp, me_sbp, mesd_sbp, rmse_sbp, mae_dbp, me_dbp, mesd_dbp, rmse_dbp))

    print(" ---------------------------------------------------------------------------------  \n")


def compute_mean_values(outputs, test_index):
    """
    计算 test_index 中相同数据的索引对应的 outputs 值的平均值。

    参数:
    outputs (torch.Tensor): 形状为 (N, 2) 的输出值张量。
    test_index (torch.Tensor): 形状为 (N, 1) 的索引值张量。

    返回:
    (torch.Tensor, torch.Tensor): 每个唯一索引及其对应的平均值。
    """
    unique_indices = torch.unique(test_index)
    mean_values = []

    for idx in unique_indices:
        if idx == 0:
            continue
        same_index_mask = (test_index.squeeze() == idx)
        outputs[same_index_mask, :] = outputs[same_index_mask, :].mean(dim=0)

    return unique_indices, outputs
