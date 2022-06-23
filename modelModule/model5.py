from json import encoder
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import restore_data

class VAE5(nn.Module):
    def __init__(self, pro_types, dim=57):
        '''
        pro_type: list, 每个元素为(x, y), x表示每个维度数据类型: normal, discrete. y表示discrete字典长度
        '''
        super(VAE5, self).__init__()
        self.dim = dim
        self.pro_types = pro_types

        self.ClassHeads = nn.ModuleList([])
        self.embeddings = nn.ModuleList([])
        for pro_type in self.pro_types:
            if pro_type[0] == 'discrete':
                self.embeddings.append(nn.Embedding(num_embeddings=pro_type[1], embedding_dim=128))
                # self.ClassHeads.append(nn.Sequential(nn.Linear(128, 128),
                #                                     nn.LeakyReLU(inplace=True),
                #                                     nn.Linear(128, pro_type[1])),
                #                                     nn.Softmax(dim=1),
                #                                     )
            elif pro_type[0] == 'normal':
                self.embeddings.append(nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1, stride=1))
                # self.ClassHeads.append(nn.Sequential(nn.Linear(128, 128),
                #                                     nn.AdaptiveMaxPool1d(output_size=1))
                #                                     )

        self.Nan_feature = nn.Parameter(torch.randn(1, 128)) # Nan特征用于替换缺失值的特征
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6) 

        self.FClayer_mu = nn.Sequential(
            nn.Linear(self.dim, self.dim*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim*2, self.dim),
            )  # 均值的输出

        self.FClayer_std = nn.Sequential(
            nn.Linear(self.dim, self.dim*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim*2, self.dim)
            )  # 方差的输出

        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1) # 全局池化

        self.decoder = nn.Sequential(
            nn.Linear(self.dim*2, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim),
            ) # 解码器


    def get_global_min_max(self, dataset):
        '''
        获取训练集正则化参数, 数据集属性类型
        '''
        self.global_max = dataset.Max_Val
        self.global_min = dataset.Min_Val
        if dataset.pro_type_file is None:
            self.pro_types = None
        else:
            self.pro_types = dataset.pro_types

    def reparameterize(self, mu, log_var):
        '''
        根据均值方差生成z
        '''
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, miss_data, M_matrix):
        '''
        miss_data: 包含缺失值的数据, (batch,dim)
        M_matrix: 缺失矩阵, (batch,dim)
        output: (batch, dim), 隐变量的均值, 隐变量的方差
        '''
        input = miss_data * M_matrix

        # 对数据进行embedding
        embedding_out = torch.tensor([])
        for i, embedding in enumerate(self.embeddings):
            if isinstance(embedding, nn.Embedding):
                embedding_out = torch.cat((embedding_out, embedding(input[:, i].long().reshape(-1,1))), dim = 1)
            else:
                embedding_out = torch.cat((embedding_out, embedding(input[:, i].reshape(-1,1,1)).permute(0, 2, 1)), dim = 1)
        # embedding_out=[batch, dim, 128]
        Miss_bool = M_matrix.unsqueeze(-1).expand(embedding_out.shape) # [batch, dim, 128]
        encoder_input = embedding_out * Miss_bool + (1 - Miss_bool) * self.Nan_feature # 将缺失数值替换为NAN

        h = self.encoder(encoder_input)              # 得到隐藏层 [batch, dim, 128]
        h = self.max_pool(h).squeeze(-1)          # 全局最大池化 [batch, dim]

        mu = self.FClayer_mu(h)       # 得到均值   [batch, dim]
        log_var = self.FClayer_std(h) # 得到方差   [batch, dim]

        z = self.reparameterize(mu, log_var) # 得到隐藏变量 [batch, dim]

        decoder_input = torch.cat(dim = -1, tensors = (z, h))        # [batch, dim]
        decoder_out = self.decoder(decoder_input)     # [batch, dim]
        out = torch.sigmoid(decoder_out)

        return out, mu, log_var


    def inference(self, miss_date, Missing):
        '''
        使用模型对缺失数据进行插补
        miss_data: 是包含nan的np数组(没有正则化的)
        Missing: 是缺失矩阵
        return: 复原后的完整数据, 模型直接输出结果
        '''
        ## 将数据正则化
        res_data = np.zeros(miss_date.shape)
        for i in range(len(miss_date)):
            if (self.pro_types is not None) and (self.pro_types[i][0] == 'discrete'):
                res_data[i,:] = miss_date[i,:]
            else:
                res_data[i,:] = (miss_date[i,:] - self.global_min) / self.global_max

        ## 将缺失部分采用999填充
        input_data = np.nan_to_num(res_data, nan=9999)
        output, _, _ = self.forward(torch.from_numpy(input_data).float(), torch.from_numpy(Missing).float())

        ## 输出完整数据
        imputed_data = restore_data(output.detach().numpy(), self.global_max, self.global_min)
        return imputed_data, output