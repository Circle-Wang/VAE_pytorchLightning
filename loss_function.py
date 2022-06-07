import torch

def vae_loss(src_data, imputed_data, M, mu, log_var):
    '''
    根据解码器输出,原始数据和缺失矩阵计算MSE, 只计算缺失部分损失
    '''
    MSE_loss = torch.sum(((1-M) * src_data - (1-M) * imputed_data)**2) / torch.sum(1-M)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    train_loss = MSE_loss + kl_div
    return train_loss
