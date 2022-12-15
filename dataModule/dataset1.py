import pickle
import torch
import numpy as np
import pandas as pd

from modelModule.utils import minmax_norm, mean_norm


class FlatDataset(torch.utils.data.Dataset):
    '''
    根据缺失数据集(.CSV文件)获得缺失矩阵, 对数据矩阵进行归一化/标准化, 返回
    '''
    def __init__(self, csv_file, pro_type_file, replace_dict_file, data_norm='minmax_norm'):
        '''
        csv_file: .csv文件路径
        data_norm: 对数据矩阵的归一化/标准化方式
        pro_type_file: pro_type.pkl文件路径
        replace_dict_file: 对离散数据集合进行替换的字典文件, key=列名,value={替换方法}

        '''
        self.csv_file = csv_file
        # self.pro_type_file = pro_type_file

        if data_norm == 'minmax_norm':
            self.data_norm = minmax_norm
        elif data_norm == 'mean_norm':
            self.data_norm = mean_norm


        self.data = pd.read_csv(csv_file)
        self.miss_matrix = 1 - self.data.isna().to_numpy().astype(int)  # 得到数据集的缺失矩阵np

        # 将缺失数据集中离散型数据，替换为从0开始的连续自然数
        replace_dict = pickle.load(open(replace_dict_file, 'rb'))
        for key, mapping in replace_dict.items():
            self.data[key] = self.data[key].map(mapping)

        # 根据pro_types继续部分正则化
        pro_types = pickle.load(open(pro_type_file, 'rb'))
        self.portion_normal_data, self.global_normal_data, self.Min_Val, self.Max_Val = self.data_norm(self.data, pro_types)

        self.data = self.data.to_numpy()  # 将原始数据集转化np


    def __len__(self):
        return len(self.data)
    
    def collater(self, samples):
        '''
        在DataLoader中参数collate_fn的传入值,主要作用对每个batch中的每个样本进行进行整合(产生Missing矩阵,产生Z加入矩阵)和规整输出
        return:字典,注意此处返回的结果就是dataloader每次返回的batch的结果
        '''
        src_data_batch = np.stack([s['src_data'] for s in samples], axis=0)              # batch中样本的原数据集
        global_normal_batch = np.stack([s['global_normal_data'] for s in samples], axis=0) 
        portion_normal_batch = np.stack([s['portion_normal_data'] for s in samples], axis=0)
        miss_matrix_batch = np.stack([s['miss_matrix'] for s in samples], axis=0)

        # 为了之后训练不出现bug, 需要将所有nan的部分变为缺失部分采用9999来补全
        # 此处并不影响模型的训练
        src_data_batch[np.isnan(src_data_batch)] = 9999
        global_normal_batch[np.isnan(global_normal_batch)] = 9999
        portion_normal_batch[np.isnan(portion_normal_batch)] = 9999

        return {
            'src_data': torch.from_numpy(src_data_batch).float(),
            'global_normal': torch.from_numpy(global_normal_batch).float(),
            'portion_normal': torch.from_numpy(portion_normal_batch).float(),
            'miss_matrix': torch.from_numpy(miss_matrix_batch).float(),
            'global_max': torch.from_numpy(self.Max_Val).float(),
            'global_min': torch.from_numpy(self.Min_Val).float(),
        }
    
    def __getitem__(self, index):
        ret = {
            'src_data': self.data[index],
            'global_normal_data': self.global_normal_data[index],
            'portion_normal_data': self.portion_normal_data[index],
            'miss_matrix': self.miss_matrix[index],
        }
            
        return ret


class ValidDataset(torch.utils.data.Dataset):
    '''
    根据缺失数据集(.CSV文件)获得缺失矩阵, 对数据矩阵进行归一化/标准化, 返回
    '''
    def __init__(self, miss_file, complete_file, pro_type_file, replace_dict_file, data_norm='minmax_norm'):
        '''
        miss_file: 缺失数据的.csv文件路径
        complete_file: 完整数据.csv文件路径
        data_norm: 对数据矩阵的归一化/标准化方式
        pro_type_file: pro_type.pkl文件路径
        replace_dict_file: 对离散数据集合进行替换的字典文件, key=列名,value={替换方法}

        '''

        if data_norm == 'minmax_norm':
            self.data_norm = minmax_norm
        elif data_norm == 'mean_norm':
            self.data_norm = mean_norm

        self.complete_data = pd.read_csv(complete_file)  # 完整数据集合
        self.miss_data = pd.read_csv(miss_file)          # 缺失数据集
        self.miss_matrix = 1 - self.miss_data.isna().to_numpy().astype(int)  # 得到数据集的缺失矩阵np


        # 将缺失数据集中离散型数据，替换为从0开始的连续自然数
        replace_dict = pickle.load(open(replace_dict_file, 'rb'))
        for key, mapping in replace_dict.items():
            self.miss_data[key] = self.miss_data[key].map(mapping)
            self.complete_data[key] = self.complete_data[key].map(mapping)

        # 根据pro_types继续部分正则化
        pro_types = pickle.load(open(pro_type_file, 'rb'))
        self.portion_normal_data, _, _, _ = self.data_norm(self.miss_data, pro_types)   # 包含缺失值和离散值 
        _, self.global_normal_data, _, _ = self.data_norm(self.complete_data, pro_types)     # 不包含缺失值和离散值
    

        # self.miss_data = self.miss_data.to_numpy()  # 将原始数据集转化np


    def __len__(self):
        return len(self.miss_data)
    
    def collater(self, samples):
        '''
        在DataLoader中参数collate_fn的传入值,主要作用对每个batch中的每个样本进行进行整合(产生Missing矩阵,产生Z加入矩阵)和规整输出
        return:字典,注意此处返回的结果就是dataloader每次返回的batch的结果
        '''
        # src_data_batch = np.stack([s['src_data'] for s in samples], axis=0)              # batch中样本的原数据集
        global_normal_batch = np.stack([s['global_normal_data'] for s in samples], axis=0) 
        portion_normal_batch = np.stack([s['portion_normal_data'] for s in samples], axis=0)
        miss_matrix_batch = np.stack([s['miss_matrix'] for s in samples], axis=0)

        portion_normal_batch[np.isnan(portion_normal_batch)] = 9999

        return {
            # 'src_data': torch.from_numpy(src_data_batch).float(),
            'global_normal': torch.from_numpy(global_normal_batch).float(),
            'portion_normal': torch.from_numpy(portion_normal_batch).float(),
            'miss_matrix': torch.from_numpy(miss_matrix_batch).float(),
            # 'global_max': torch.from_numpy(self.Max_Val).float(),
            # 'global_min': torch.from_numpy(self.Min_Val).float(),
        }
    
    def __getitem__(self, index):
        ret = {
            # 'src_data': self.data[index],
            'global_normal_data': self.global_normal_data[index],
            'portion_normal_data': self.portion_normal_data[index],
            'miss_matrix': self.miss_matrix[index],
        }
            
        return ret