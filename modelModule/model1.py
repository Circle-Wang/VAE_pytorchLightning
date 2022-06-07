import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, dim=57):
        super(VAE, self).__init__()
        self.dim = dim
        self.FClayer1 = nn.Linear(in_features=self.dim*2, out_features=self.dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=3, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6) # 输出为[batch, src, dim]
        self.FClayer_mu = nn.Linear(in_features=self.dim, out_features=30) # 均值的输出[batch, src, dim]
        self.FClayer_std = nn.Linear(in_features=self.dim, out_features=30) # 方差的输出

        
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=30, nhead=6, dim_feedforward=256, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=6)
        self.FClayer2 = nn.Linear(in_features=30, out_features=self.dim) # 把解码器得到的数据变成我们需要的数据



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
        input = self.FClayer1(input) # 将缺失矩阵和缺失数据联系起来

        h = self.encoder(input)
        mu = self.FClayer_mu(h) # 得到均值
        log_var = self.FClayer_std(h) # 得到方差

        z = self.reparameterize(mu, log_var) # 得到隐藏变量

        out = self.decoder(z)
        out = torch.sigmoid(self.FClayer2(out)).squeeze(0) # [batch,dim]
        return out, mu, log_var


