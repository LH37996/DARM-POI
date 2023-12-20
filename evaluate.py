import os
import json
import numpy as np
import datetime
from sklearn import metrics
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

dataset='florida' ##################### modify the dataset here
filepath='./data/data_{}/'.format(dataset)
resultpath = './output/output_{}/'.format(dataset)
assert os.path.exists(resultpath)


class MaximumMeanDiscrepancy_numpy(object):
    """calculate MMD"""

    def __init__(self):
        super(MaximumMeanDiscrepancy_numpy, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = source.shape[0]+ target.shape[0]
        total = np.concatenate([source, target], axis=0)  # 合并在一起
        total0 = np.expand_dims(total, axis = 0)
        total0 = np.tile(total0, (total.shape[0], 1, 1))

        total1 = np.expand_dims(total, axis = 1)
        total1 = np.tile(total1, (1, total.shape[0], 1))

        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [np.exp(-L2_distance / bandwidth_temp) for
                      bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 将多个核合并在一起

    def __call__(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = source.shape[0]
        m = target.shape[0]

        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        # K_ss矩阵，Source<->Source
        XX = np.divide(XX, n * n).sum(axis=1).reshape(1, -1)
        # K_st矩阵，Source<->Target
        XY = np.divide(XY, -n * m).sum(axis=1).reshape(1, -1)

        # K_ts矩阵,Target<->Source
        YX = np.divide(YX, -m * n).sum(axis=1).reshape(1, -1)
        # K_tt矩阵,Target<->Target
        YY = np.divide(YY, m * m).sum(axis=1).reshape(1, -1)

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss


file_path = 'data/data_florida/aggregated_florida_visits.csv'
florida_visits_df = pd.read_csv(file_path)

def extract_monthly_data(df, year, months):
    """
    Extracts data for specific months of a given year from a dataframe.

    :param df: DataFrame containing the data.
    :param year: The specific year (e.g., 2019).
    :param months: List of months (integers) to extract data for.
    :return: A 2D list where each outer list item is a row from the dataframe,
             and the inner list contains data for the specified months.
    """
    # Construct column names for the specified months
    month_columns = [f"{year}-{str(month).zfill(2)}" for month in months]

    # Extract the relevant columns
    extracted_data = df[month_columns].values.tolist()

    return extracted_data

extracted_2019_data = extract_monthly_data(florida_visits_df, 2019, [9, 10, 11, 12])
socall_nreg = len(extracted_2019_data)
three_dim_list = []
for i in range(20):
    three_dim_list.append(extracted_2019_data)
def to_four_dimensions(three_dim_list):
        # 通过列表推导式遍历每个元素，并将其转换成一个新的列表
        return [[[[element] for element in inner_list] for inner_list in outer_list] for outer_list in
                three_dim_list]


allday_data = to_four_dimensions(three_dim_list)

print(allday_data)

allday_data=np.array(allday_data)

M=np.max(allday_data)
m=np.min(allday_data)

# allday_data = (2 * allday_data - m - M) / (M - m)  # 归一化到 [-1, 1]

allday_flow=np.mean(allday_data,0)

# load train and test regions
with open(filepath + 'train_regs.json', 'r') as f:
    trainregs = json.load(f)
with open(filepath + 'test_regs.json', 'r') as f:
    testregs = json.load(f)

trainids = [x for x in trainregs]
sampids = [x for x in testregs]

test_reg_flow = allday_flow[sampids,:,:]
test_flow = allday_data[:,sampids,:,:]


def cal_smape(p_pred, p_real, eps=0.00000001):
    out=np.mean(np.abs(p_real - p_pred) / ((np.abs(p_real) + np.abs(p_pred)) / 2 + eps))
    return out

it=500
pred=np.load(resultpath+"sample_{}_final.npz".format(it))

pred=pred['sample']
pred=(pred*(M-m)+m+M)/2

pred_flow=np.mean(pred,0)
rmse=metrics.mean_squared_error(pred_flow.flatten(),test_reg_flow.flatten(),squared=False)
mae=metrics.mean_absolute_error(pred_flow.flatten(),test_reg_flow.flatten())
smape=cal_smape(pred_flow.flatten(),test_reg_flow.flatten())

mmd = MaximumMeanDiscrepancy_numpy()
tmpmmds=[]
for i in range(pred.shape[1]):
    realflow=test_flow[:,i,:,:]
    genflow=pred[:,i,:,:]
    data_1=realflow.reshape(realflow.shape[0],-1)
    data_2=genflow.reshape(genflow.shape[0],-1)
    tmpmmds.append(mmd(data_1, data_2))
mmd=np.mean(tmpmmds)
print('%.2f\t%.2f\t%.2f\t%.2f'%(mae,rmse,smape,mmd))
