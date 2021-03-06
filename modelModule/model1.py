import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import restore_data

class VAE(nn.Module):
    def __init__(self, dim=57, nhead=3):
        super(VAE, self).__init__()
        self.dim = dim
        self.nhead = nhead
        # self.scale_parm = nn.Parameter(torch.randn(1, self.dim))
        # self.FClayer1 = nn.Linear(in_features=self.dim*2, out_features=self.dim)
        self.FClayers1 = nn.Sequential(
            nn.Linear(self.dim*2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, self.dim),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6) # 输出为[batch, src, dim]
        self.FClayer_mu = nn.Linear(in_features=self.dim, out_features=30)  # 均值的输出[batch, src, dim]
        self.FClayer_std = nn.Linear(in_features=self.dim, out_features=30) # 方差的输出

        self.decoder_layer = nn.TransformerEncoderLayer(d_model=30, nhead=6, dim_feedforward=256, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=6)
        self.FClayer2 = nn.Linear(in_features=30, out_features=self.dim)   # 把解码器得到的数据变成我们需要的数据

    def get_global_min_max(self, global_max, global_min):
        '''
        获取训练集正则化参数
        '''
        self.global_max = global_max
        self.global_min = global_min

    def reparameterize(self, mu, log_var):
        '''
        根据均值方差生成z
        '''
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, miss_data, M_matrix):
        '''
        miss_data: 包含缺失值的数据，(batch,dim)
        M_matrix: 缺失矩阵
        output: (batch, dim), 隐变量的均值, 隐变量的方差
        '''
        input = torch.cat(dim = 1, tensors = (miss_data, M_matrix)).unsqueeze (0) # [1, batch, dim]
        input = self.FClayers1(input) # 将缺失矩阵和缺失数据联系起来

        h = self.encoder(input) # 得到隐藏层
        mu = self.FClayer_mu(h) # 得到均值
        log_var = self.FClayer_std(h) # 得到方差

        z = self.reparameterize(mu, log_var) # 得到隐藏变量

        out = self.decoder(z)
        out = torch.sigmoid(self.FClayer2(out)).squeeze(0) # [batch, dim]
        # out = out * self.scale_parm
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
            res_data[i,:] = (miss_date[i,:] - self.global_min) / self.global_max

        ## 将缺失部分采用999填充
        input_data = np.nan_to_num(res_data, nan=9999)
        output, _, _ = self.forward(torch.from_numpy(input_data).float(), torch.from_numpy(Missing).float())

        ## 输出完整数据
        imputed_data = restore_data(output.detach().numpy(), self.global_max, self.global_min)
        return imputed_data, output
