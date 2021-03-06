import pickle
import torch
import numpy as np
from utils import get_missing, minmax_norm, mean_norm


class FlatDataset(torch.utils.data.Dataset):
    '''
    根据完整数据,得到缺失矩阵, 并对矩阵进行[0,1)区间的标准化,返回不含随机数的缺失数组
    '''
    def __init__(self, csv_file, missing_ratio=0.3, data_norm='minmax_norm', pro_type_file=None):
        self.csv_file = csv_file
        self.missing_ratio = missing_ratio
        self.pro_type_file = pro_type_file # 数据属性文件
        if data_norm == 'minmax_norm':
            self.data_norm = minmax_norm
        elif data_norm == 'mean_norm':
            self.data_norm = mean_norm

        self.data = np.loadtxt(self.csv_file, delimiter=",", skiprows=1)

        if pro_type_file is None:
            self.global_normal_data, self.Min_Val, self.Max_Val = self.data_norm(self.data)
        else:
            self.pro_types = pickle.load(open(pro_type_file, 'rb'))
            self.portion_normal_data, self.global_normal_data, self.Min_Val, self.Max_Val = self.data_norm(self.data, self.pro_types)


    def __len__(self):
        return len(self.data)
    
    def collater(self, samples):
        '''
        在DataLoader中参数collate_fn的传入值,主要作用对每个batch中的每个样本进行进行整合(产生Missing矩阵,产生Z加入矩阵)和规整输出
        return:字典,注意此处返回的结果就是dataloader每次返回的batch的结果
        '''
        src_data_batch = np.stack([s['src_data'] for s in samples], axis=0)       # batch中样本的原数据集
        normal_data_batch = np.stack([s['global_normal_data'] for s in samples], axis=0)
        if self.pro_type_file is not None:
            portion_normal_data = np.stack([s['portion_normal_data'] for s in samples], axis=0)
            miss_data_batch, M_batch = get_missing(portion_normal_data, self.missing_ratio)   # 得到缺失矩阵
        else:
            miss_data_batch, M_batch = get_missing(normal_data_batch, self.missing_ratio)   # 得到缺失矩阵
        # miss_data:缺失部分采用9999来补全

        return {
            "src_data": torch.from_numpy(src_data_batch).float(),
            'normal_data': torch.from_numpy(normal_data_batch).float(),
            "miss_data": torch.from_numpy(miss_data_batch).float(), 
            'miss_matrix': torch.from_numpy(M_batch).float(),
            'global_max': torch.from_numpy(self.Max_Val).float(),
            'global_min': torch.from_numpy(self.Min_Val).float(),
        }
    
    def __getitem__(self, index):
        if self.pro_type_file is None:
            ret = {
                'src_data': self.data[index],
                'global_normal_data': self.global_normal_data[index],
            }
        else:
            ret = {
                'src_data': self.data[index],
                'global_normal_data': self.global_normal_data[index],
                'portion_normal_data': self.portion_normal_data[index],
            }
            
        return ret

