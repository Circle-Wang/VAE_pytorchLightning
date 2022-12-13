from json import encoder
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .utils import restore_data, minmax_norm

class VAE(nn.Module):
    def __init__(self, pro_types, replace_dict, dim):
        '''
        pro_type: list, 每个元素为(x, y), x表示每个维度数据类型: normal, discrete. y表示discrete字典长度
        replace_dict_file: 对离散数据集合进行替换的字典文件, key=列名,value={替换方法}
        '''
        super(VAE, self).__init__()
        self.dim = dim
        self.pro_types = pro_types
        self.replace_dict = replace_dict  # 用于推理和复原文件

        self.attribute_type = []  # 表示1表示连续型，0表示离散型, 表示每一个属性是离散型还是连续型
        # self.ClassHeads = nn.ModuleList([])
        self.embeddings = nn.ModuleList([])
        for pro_type in self.pro_types:
            if pro_type[0] == 'discrete':
                self.embeddings.append(nn.Embedding(num_embeddings=pro_type[1], embedding_dim=64))
                # self.ClassHeads.append(nn.Sequential(nn.Linear(128, 128),
                #                                     nn.LeakyReLU(inplace=True),
                #                                     nn.Linear(128, pro_type[1])),
                #                                     nn.Softmax(dim=1),
                #                                     )
                self.attribute_type.append(0)
            elif pro_type[0] == 'normal':
                self.embeddings.append(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=1))
                # self.ClassHeads.append(nn.Sequential(nn.Linear(128, 128),
                #                                     nn.AdaptiveMaxPool1d(output_size=1))
                #                                     )
                self.attribute_type.append(1)

        self.Nan_feature = nn.Parameter(torch.randn(1, 64)) # Nan特征用于替换缺失值的特征
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6) 


        # # 使用max_pool的decoder
        # self.FClayer_mu = nn.Sequential(
        #     nn.Linear(self.dim, self.dim*2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(self.dim*2, self.dim),
        #     )  # 均值的输出

        # self.FClayer_std = nn.Sequential(
        #     nn.Linear(self.dim, self.dim*2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(self.dim*2, self.dim)
        #     )  # 方差的输出

        # # self.max_pool = nn.AdaptiveMaxPool1d(output_size=1) # 全局最大池化,全局平均池化nn.AdaptiveAvgPool1d(output_size=1)
        # self.mean_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.dim*2, self.dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.dim, self.dim),
        #     ) # 解码器

        # 不使用max_pool的decoder
        self.FClayer_mu = nn.Sequential(
            nn.Linear(64, 64*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*2, 64),
            )  # 均值的输出

        self.FClayer_std = nn.Sequential(
            nn.Linear(64, 64*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*2, 64),
            )  # 方差的输出

        # self.max_pool = nn.AdaptiveMaxPool1d(output_size=1) # 全局最大池化,全局平均池化nn.AdaptiveAvgPool1d(output_size=1)
        self.mean_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.decoder = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.dim),
            nn.Linear(64, 64),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(self.dim),
            # nn.Linear(128, 128),
            ) # 解码器

        ## 采用Transformer作为解码器
        # self.decoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        # self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=6)
        # self.mean_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # self.FClayers3 = nn.Sequential(
        #     nn.Linear(self.dim*2, self.dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(self.dim, self.dim),
        #     )

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
        return:
            out: (batch, dim), 
            D_tensor_list: list, 每个元素是个tensor表示离散型属性的标签
            mu: (batch, dim, 128), 隐变量的均值
            log_var: (batch, dim, 128), 隐变量的方差
        '''
        input = miss_data * M_matrix
        
        # 对数据进行embedding
        embedding_out = torch.tensor([], device=input.device)
        
        for i, embedding in enumerate(self.embeddings):
            if isinstance(embedding, nn.Embedding):
                embedding_out = torch.cat((embedding_out, embedding(input[:, i].long().reshape(-1,1))), dim = 1)
                # print(f"=========={i}=============")
                # print("权重: ", embedding.weight)
                # print("梯度: ", embedding.weight.grad)
            else:
                embedding_out = torch.cat((embedding_out, embedding(input[:, i].reshape(-1,1,1)).permute(0, 2, 1)), dim = 1)
        # embedding_out=[batch, dim, 128]
        Miss_bool = M_matrix.unsqueeze(-1).expand(embedding_out.shape) # [batch, dim, 64]
        encoder_input = embedding_out * Miss_bool + (1 - Miss_bool) * self.Nan_feature # 将缺失数值替换为NAN
    
        h = self.encoder(encoder_input)              # 得到隐藏层 [batch, dim, 64]
        
        mu = self.FClayer_mu(h)       # 得到均值   [batch, dim, 64]
        log_var = self.FClayer_std(h) # 得到方差   [batch, dim, 64]

        z = self.reparameterize(mu, log_var) # 得到隐藏变量 [batch, dim, 64]

        decoder_input = torch.cat(dim = -1, tensors = (z, h))        # [batch, dim, 64*2]
        decoder_out = self.decoder(decoder_input)     # [batch, dim, 64]

        D_tensor_list = []

        for i, embedding in enumerate(self.embeddings):
            if isinstance(embedding, nn.Embedding):
                d_tensor = torch.mm(decoder_out[:,i,:], embedding.weight.T) # [batch, class_number]
                D_tensor_list.append(d_tensor)
            else:
                D_tensor_list.append(None)

        out = self.mean_pool(decoder_out).squeeze(-1) # 全局平均池化 [batch, dim]
        out = torch.sigmoid(out)    # [batch, dim]
        mask_attribute_type = torch.tensor(self.attribute_type, device=input.device)  # 表示1表示连续型，0表示离散型 [dim]
        out = out * mask_attribute_type.unsqueeze(0).expand(miss_data.shape)  # [batch, dim]

        return out, D_tensor_list, mu, log_var

    # #### 不使用max_pool
    # def forward(self, miss_data, M_matrix):
    #     '''
    #     miss_data: 包含缺失值的数据, (batch,dim)
    #     M_matrix: 缺失矩阵, (batch,dim)
    #     output: (batch, dim), 隐变量的均值, 隐变量的方差
    #     '''
    #     input = miss_data * M_matrix

    #     # 对数据进行embedding
    #     embedding_out = torch.tensor([])
    #     for i, embedding in enumerate(self.embeddings):
    #         if isinstance(embedding, nn.Embedding):
    #             embedding_out = torch.cat((embedding_out, embedding(input[:, i].long().reshape(-1,1))), dim = 1)
    #         else:
    #             embedding_out = torch.cat((embedding_out, embedding(input[:, i].reshape(-1,1,1)).permute(0, 2, 1)), dim = 1)
    #     # embedding_out=[batch, dim, 128]
    #     Miss_bool = M_matrix.unsqueeze(-1).expand(embedding_out.shape) # [batch, dim, 128]
    #     encoder_input = embedding_out * Miss_bool + (1 - Miss_bool) * self.Nan_feature # 将缺失数值替换为NAN

    #     h = self.encoder(encoder_input)              # 得到隐藏层 [batch, dim, 128]

    #     mu = self.FClayer_mu(h)       # 得到均值   [batch, dim, 128]
    #     log_var = self.FClayer_std(h) # 得到方差   [batch, dim, 128]

    #     z = self.reparameterize(mu, log_var) # 得到隐藏变量 [batch, dim, 128]

    #     decoder_input = torch.cat(dim = 1, tensors = (z, h))        # [batch, dim*2, 128]
    #     decoder_out = self.decoder(decoder_input)     # [batch, dim*2, 128]
    #     layer3_input = self.mean_pool(decoder_out).squeeze(-1)  # [batch, dim*2]
    #     layer3_out = self.FClayers3(layer3_input)
    #     out = torch.sigmoid(layer3_out)
    #     return out, mu, log_var


    def inference(self, miss_date, restore=True):
        '''
        使用模型对缺失数据进行插补
        miss_data: 是包含nan的DF(文件)
        return: 复原后的完整数据, 模型直接输出结果
        '''
        ## 将数据进行处理，替换离散数据集合
        miss_date_copy = miss_date.copy()
        for key, mapping in self.replace_dict.items():
            miss_date_copy[key] = miss_date_copy[key].map(mapping)

        Missing = 1 - miss_date_copy.isna().to_numpy().astype(int) # 获取缺失矩阵np
        
        ## 将数据正则化
        partial_norm_data, _, Min_Val, Max_Val = minmax_norm(miss_date_copy, self.pro_types)

        ## 将缺失部分采用999填充
        input_data = np.nan_to_num(partial_norm_data, nan=999)

        with torch.no_grad():
            output, D_tensor_list, _, _ = self.forward(torch.from_numpy(input_data).float(), torch.from_numpy(Missing).float())

        ## 还原最大最小值
        imputed_data = output * (torch.from_numpy(Max_Val).float() - torch.from_numpy(Min_Val).float()) + torch.from_numpy(Min_Val).float() # [batch, dim]恢复原来的值
        for index, pro_type in enumerate(self.attribute_type):
            if pro_type == 0:
                imputed_data[:, index] = torch.argmax(D_tensor_list[index], dim=1)


        ## 根据列最大最小将数据进行复原
        imputed_data = imputed_data.detach().numpy() * (1-Missing) + Missing * np.nan_to_num(miss_date_copy, nan=999) # 先将miss_data中的nan换为99 防止计算无效
        imputed_data = pd.DataFrame(imputed_data, columns=miss_date_copy.columns)

        ## 根据离散数据映射表，将数据复原
        new_dict = dict()  ## 得到反向映射字典
        for key, value_dict in self.replace_dict.items():
            new_dict[key] = dict(zip(value_dict.values(), value_dict.keys()))
        for key, mapping in new_dict.items():
            imputed_data[key] = imputed_data[key].map(mapping)

        return imputed_data