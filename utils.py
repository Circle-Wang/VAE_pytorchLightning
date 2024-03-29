import numpy as np
import torch

def minmax_norm(src_data, pro_types=None):
    '''
    对数据进行按列进行最大-最小正则化(减去最小值除以最大值),使得每一个数据都处于[0,1]区间
    pro_types: List 元素为元组, 每个元组第一个位置表示该属性类别discrete, 或者normal
    返回值
    '''
    data = src_data.copy().astype(np.float32)
    num, Dim = data.shape
    # 记录各列最大值,最小值用于返回得到插值结果
    Min_Val = np.zeros(Dim) 
    Max_Val = np.zeros(Dim)
    for i in range(Dim):
        Min_Val[i] = np.min(src_data[:,i])
        Max_Val[i] = np.max(src_data[:,i])
        data[:,i] = (src_data[:,i] - np.min(src_data[:,i]))/(np.max(src_data[:,i]) - np.min(src_data[:,i]) + 1e-6) # 减去最小值

    if pro_types is None:
        return data, Min_Val, Max_Val
    else:
        model_data = data.copy()
        for i in range(Dim):
            if pro_types[i][0] == 'discrete':
                model_data[:,i] = src_data[:,i]
        return model_data, data, Min_Val, Max_Val


def mean_norm(src_data, pro_types=None):
    '''
    对数据进行按列进行均值-标准差正则化(减去均值除以方差), 使得每一个数据都处于[0,1]区间
    '''
    data = src_data.copy().astype(np.float32)
    num, Dim = data.shape
    # 记录各列最大值,最小值用于返回得到插值结果
    mean_Val = np.zeros(Dim) 
    std_Val = np.zeros(Dim)
    for i in range(Dim):
        mean_Val[i] = np.mean(src_data[:,i])
        std_Val[i] = np.std(src_data[:,i])
        data[:,i] = src_data[:,i] - np.mean(src_data[:,i]) / (np.std(src_data[:,i]) + 1e-8)  # 减去最小值

    if pro_types is None:
        return data, mean_Val, std_Val
    else:
        model_data = data.copy()
        for i in range(Dim):
            if pro_types[i][0] == 'discrete':
                model_data[:,i] = src_data[:,i]
        return model_data, data, mean_Val, std_Val


def get_missing(data, p_miss):
    '''
    得到缺失缺失矩阵(1代表存在数据,0代表缺失数据), 以及包含缺失数据的数据集, 缺失数据采用9999替代
    data: 完整数据
    p: 缺失概率
    return: 包含缺失数据的矩阵, Missing矩阵
    '''
    num, Dim = data.shape
    p_miss_vec = p_miss * np.ones((Dim,1))
    Missing = np.zeros((num, Dim)) # 缺失矩阵, 1代表存在数据,0代表缺失数据
    for i in range(Dim):
        A = np.random.uniform(0., 1., size = [num,]) # 从[0,1)抽取随机数，shape=size，此处为(4601,)
        B = A > p_miss_vec[i] # (4601,)返回bool向量，如果随机数大于p_miss_vec则为1，控制缺失比率
        Missing[:,i] = 1.*B   # 得到随机缺失矩阵
    
    missing_data = data * Missing + 9999 * (1-Missing)
    return missing_data, Missing

def restore_data(data, max_val, min_val, norm_type="minmax_norm"):
    '''
    根据列向量(属性)的最大值(标准差)和最小值(均值), 复原真正的数据,
    '''
    if isinstance(data, np.ndarray):
        res_data = np.zeros(data.shape)
    else:
        res_data = torch.zeros_like(data, device=data.device)

    if norm_type == "minmax_norm":
        for i in range(len(data)):
            res_data[i,:] = data[i,:] * (max_val-min_val) + min_val
    elif norm_type == "mean_norm":
        for i in range(len(data)):
            res_data[i,:] = data[i,:] * max_val + min_val
    return res_data

def result_show(src_data, imputed_data, M_matrix):
    '''
    根据原来数据, 插补数据, 以及缺失矩阵计算缺失部分的插补结果展示。
    src_data: (batch, dim)输入维度
    '''
    src_imputed = src_data[M_matrix==0]
    gen_imputed = imputed_data[M_matrix==0]
    print("原始数据",src_imputed)
    print("插补数据",gen_imputed)
    diff_ratio = abs(src_imputed - gen_imputed)/(src_imputed+1e-8)
    return diff_ratio