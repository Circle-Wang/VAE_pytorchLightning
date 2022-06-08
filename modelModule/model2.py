import torch
from torch import nn
import torch.nn.functional as F

class VAE2(nn.Module):
    def __init__(self, dim=57, nhead=3):
        super(VAE2, self).__init__()
        self.dim = dim
        self.nhead = nhead
        self.FClayers1 = nn.Sequential(
            nn.Linear(self.dim*2, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.dim),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.nhead, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6) # 输出为[batch, src, dim]

        self.FClayer_mu = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            )  # 均值的输出[batch, src, dim]
        self.FClayer_std = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            )  # 方差的输出

        self.FClayers2 = nn.Sequential(
            nn.Linear(self.dim+128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            ) # 用于连接编码器输出变量h以及生成的随机数z

        self.decoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=6)
        self.FClayers3 = nn.Sequential(
            nn.Linear(128, self.dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
            ) # 把解码器得到的数据变成我们需要的数据


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

        z = self.reparameterize(mu, log_var) # 得到隐藏变量 [1, batch, 128]

        out = self.FClayers2(torch.cat(dim = -1, tensors = (z, h)))
        out = self.decoder(out)
        out = torch.sigmoid(self.FClayers3(out)).squeeze(0) # [batch, dim]
        # out = out * self.scale_parm
        return out, mu, log_var


