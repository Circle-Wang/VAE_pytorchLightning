import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import restore_data

class VAE3(nn.Module):
    def __init__(self, dim=57, nhead=3):
        super(VAE3, self).__init__()
        self.dim = dim
        self.nhead = nhead
    
        self.FClayers1 = nn.Sequential(
            nn.Linear(self.dim*2, self.dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
        )
        self.embedding1 = nn.Linear(1, 128)
        self.embedding2 = nn.Linear(1, 128)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6) # 输出为[batch, 256, 128]

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

        self.max_pool1 = nn.AdaptiveMaxPool1d(output_size=1) # 全局池化
        self.max_pool2 = nn.AdaptiveMaxPool1d(output_size=1) # 全局池化 

        self.FClayers2 = nn.Sequential(
            nn.Linear(self.dim*2, self.dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
            )

        self.decoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=6)
        self.FClayers3 = nn.Sequential(
            nn.Linear(self.dim, self.dim*4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim*4, self.dim),
            ) # 把解码器得到的数据变成我们需要的数据

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
        input = torch.cat(dim = 1, tensors = (miss_data, M_matrix))
        input = self.embedding1(self.FClayers1(input).unsqueeze (-1))  # 将缺失矩阵和缺失数据联系起来 [batch, dim, 128]

        h = self.encoder(input) # 得到隐藏层 [batch, dim, 128]
        h = self.max_pool1(h).squeeze(-1)    # 全局最大池化 [batch, dim]

        mu = self.FClayer_mu(h)       # 得到均值   [batch, dim]
        log_var = self.FClayer_std(h) # 得到方差   [batch, dim]

        z = self.reparameterize(mu, log_var) # 得到隐藏变量 [batch, dim]

        decoder_input = torch.cat(dim = 1, tensors = (z, M_matrix)) # [batch, dim*2]
        decoder_input = self.embedding2(self.FClayers2(decoder_input).unsqueeze (-1))

        decoder_out = self.decoder(decoder_input)     # [batch, dim, 128]
        out = self.max_pool2(decoder_out).squeeze(-1) # [batch, dim]
        out = self.FClayers3(out)

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
        




