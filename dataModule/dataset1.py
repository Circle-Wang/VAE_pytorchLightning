import torch
import numpy as np
from utils import get_missing, minmax_norm, mean_norm


class FlatDataset(torch.utils.data.Dataset):
    '''
    根据完整数据,得到缺失矩阵, 并对矩阵进行[0,1)区间的标准化,返回不含随机数的缺失数组
    '''
    def __init__(self, csv_file, missing_ratio=0.3, is_test=False, data_norm='minmax_norm'):
        self.csv_file = csv_file
        self.missing_ratio = missing_ratio
        self.is_test = is_test
        if data_norm == 'minmax_norm':
            self.data_norm = minmax_norm
        elif data_norm == 'mean_norm':
            self.data_norm = mean_norm

        self.data = np.loadtxt(self.csv_file, delimiter=",", skiprows=1)
        self.global_normal_data, self.Min_Val, self.Max_Val = self.data_norm(self.data)

        # if self.is_test:
        #     self.normal_data, self.Min_Val, self.Max_Val = self.data_norm(self.data)
        #     self.missing_data, self.Missing = get_missing(self.normal_data, missing_ratio) # 列缺失率为0.3

    def __len__(self):
        return len(self.data)
    
    def collater(self, samples):
        '''
        在DataLoader中参数collate_fn的传入值,主要作用对每个batch中的每个样本进行进行整合(产生Missing矩阵,产生Z加入矩阵)和规整输出
        return:字典,注意此处返回的结果就是dataloader每次返回的batch的结果
        '''
        src_data_batch = np.stack([s['src_data'] for s in samples], axis=0)       # batch中样本的原数据集
        normal_data_batch = np.stack([s['global_normal_data'] for s in samples], axis=0)

        miss_data_batch, M_batch = get_missing(normal_data_batch, self.missing_ratio)   # 得到缺失矩阵
        # miss_data_batch = np.stack([s['miss_data'] for s in samples], axis=0)     # batch中样本的缺失数据集合
        # M_batch = np.stack([s['miss_matrix'] for s in samples], axis=0) # batch中样本的缺失矩阵

        return {
            "src_data": torch.from_numpy(src_data_batch).float(),
            'normal_data': torch.from_numpy(normal_data_batch).float(),
            "miss_data": torch.from_numpy(miss_data_batch).float(),
            'miss_matrix': torch.from_numpy(M_batch).float(),
            # 'Min_Val': torch.from_numpy(Min_Val_batch).float(),
            # 'Max_Val': torch.from_numpy(Max_Val_batch).float(),
        }
    
    def __getitem__(self, index):
        ret = {
            'src_data': self.data[index],
            'global_normal_data': self.global_normal_data[index],
            # 'global_miss_data': self.global_missing_data[index],
            # 'global_miss_matrix': self.global_Missing[index],
        }
        return ret

