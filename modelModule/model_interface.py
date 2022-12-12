import pytorch_lightning as pl
import torch
from torch import nn

from .model import VAE
import pickle
from .loss_function import vae_loss

import warnings
import logging
# 忽略警告
warnings.filterwarnings("ignore")
# 初始化日志函数
logger = logging.getLogger(__name__)

class MInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        logger.info('VAE 模型初始化开始...')
        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.lr
        if self.args.model_type == 'model':
            pro_types = pickle.load(open(self.args.pro_type_file, 'rb'))
            replace_dict = pickle.load(open(self.args.replace_dict_file, 'rb'))
            self.model = VAE(dim=self.args.dim, pro_types=pro_types, replace_dict=replace_dict)

        ## 参数初始化
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def training_step(self, batch, batch_idx):
        _, portion_normal = batch['global_normal'], batch['portion_normal']
        
        M_matrix = batch['miss_matrix']
        
        
        out, D_tensor_list, mu, log_var = self.model(portion_normal, M_matrix) # [batch, dim]
        
        kl_div_loss = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        MSE_loss, EntropyLoss = vae_loss(portion_normal, out, M_matrix, D_tensor_list, self.model.attribute_type)

        loss = kl_div_loss + MSE_loss + EntropyLoss

        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('kl_div',kl_div_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('MSE_loss', MSE_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('EntropyLoss', EntropyLoss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, portion_normal = batch['global_normal'], batch['portion_normal']
        M_matrix = batch['miss_matrix']

        out, D_tensor_list, mu, log_var = self.model(portion_normal, M_matrix) # [batch, dim]
        MSE_loss, EntropyLoss = vae_loss(portion_normal, out, 1-M_matrix, D_tensor_list, self.model.attribute_type)
        val_loss = MSE_loss + EntropyLoss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_MSE_loss', MSE_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_EntropyLoss', EntropyLoss, on_epoch=True, prog_bar=True, logger=True)



    ## 优化器配置
    def configure_optimizers(self):
        logger.info('configure_optimizers 初始化开始...')
        # 选择优化器
        if self.args.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # 选择学习率调度方式
        if self.args.lr_scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=0.0002,
                                                            verbose=True,
                                                            epochs=500,
                                                            steps_per_epoch=7)
            logger.info('configure_optimizers 初始化结束...')
            return [optimizer], [scheduler]
        elif self.args.lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.args.T_max,
                                                                   eta_min=self.args.min_lr,
                                                                   verbose=True,
                                                                   last_epoch=-1)
            logger.info('configure_optimizers 初始化结束...')
            return [optimizer], [scheduler]

        elif self.args.lr_scheduler == 'None':
            logger.info('configure_optimizers 初始化结束...')
            return optimizer


