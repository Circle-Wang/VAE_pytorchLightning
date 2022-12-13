import torch
from torch import nn


def vae_loss(src_data, out, M, tensor_list, attribute_type):
    '''
    根据解码器输出, 计算原始数据和有准确值的数据计算MSE, 只计算非缺失部分损失
    src_data: (batch, dim) 部分归一化后的缺失数据集
    out: (batch, dim) 模型中最后的输出
    tensor_list: list, 其中每个元素是离散属性的模型输出(batch,dim), 用于计算交叉熵
    attribute_type: 表示1表示连续型, 0表示离散型, 表示每一个属性是离散型还是连续型
    M: 是src_data中表明缺失的部分, 1表示存在, 0表示缺失
    '''

    mask_attribute_type = torch.tensor(attribute_type, device=out.device).unsqueeze(0).expand(out.shape)  # (batch, dim)

    # 对缺失数据进行掩码处理
    src_data = src_data * M
    out = out * M 

    loos_fun = torch.nn.CrossEntropyLoss(reduction='none')
    EntropyLoss = 0
    MSE_loss = 0
    for index, pro_type in enumerate(attribute_type):
        if pro_type == 0:
            a = loos_fun(tensor_list[index], src_data[:,index].long())
            EntropyLoss = EntropyLoss + (a * M[:,index]).sum() / (M[:,index].sum())

    ## 如果M * mask_attribute_type不全为0，则计算下面MSE
    if (M * mask_attribute_type).sum() != 0:
        MSE_loss = ((src_data * mask_attribute_type - out)**2).sum() / (M * mask_attribute_type).sum()

    # MSE_loss = ((src_data * mask_attribute_type - out)**2).sum(1).mean()


    # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # train_loss = MSE_loss + kl_div
    # return train_loss, MSE_loss, kl_div
    return  MSE_loss, EntropyLoss



# def vae_loss(src_data, imputed_data, M, mu, log_var):
#     '''
#     根据解码器输出,原始数据和缺失矩阵计算MSE, 只计算缺失部分损失
#     '''
#     MSE_loss = torch.sum(((1-M) * src_data - (1-M) * imputed_data)**2) / torch.sum(1-M)
#     kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#     train_loss = MSE_loss*1000 + kl_div
#     return train_loss, MSE_loss*1000, kl_div

