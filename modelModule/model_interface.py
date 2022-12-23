import pytorch_lightning as pl
import torch
from torch import nn
import pickle

from .model1 import VAE1
from .model5 import VAE5
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
        if self.args.model_type == 'model5':
            pro_types = pickle.load(open(self.args.pro_type_file, 'rb'))
            replace_dict = pickle.load(open(self.args.replace_dict_file, 'rb'))
            self.model = VAE5(dim=self.args.dim, pro_types=pro_types, replace_dict=replace_dict)
        elif self.args.model_type == 'model1':
            pro_types = pickle.load(open(self.args.pro_type_file, 'rb'))
            replace_dict = pickle.load(open(self.args.replace_dict_file, 'rb'))
            self.model = VAE1(dim=self.args.dim, pro_types=pro_types, replace_dict=replace_dict)

        ## 参数初始化
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def training_step(self, batch, batch_idx):
        global_normal, portion_normal = batch['global_normal'], batch['portion_normal']
        M_matrix = batch['miss_matrix']
        # src_data, global_normal, portion_normal = batch['src_data'], batch['global_normal'], batch['portion_normal']
        # M_matrix = batch['miss_matrix']
        # global_max, global_min = batch['global_max'], batch['global_min']

        imputed_data, mu, log_var = self.model(portion_normal, M_matrix) # [batch, dim]

        loss, MSE_loss, kl_div = vae_loss(global_normal, imputed_data, M_matrix, mu, log_var)

        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('kl_div', kl_div, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('MSE_loss', MSE_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        global_normal, portion_normal = batch['global_normal'], batch['portion_normal']
        M_matrix = batch['miss_matrix']
        # global_max, global_min = batch['global_max'], batch['global_min']

        imputed_data, mu, log_var = self.model(portion_normal, M_matrix) # [batch, dim]
        
        # imputed_data = imputed_data * global_max + global_min # 恢复原来的值
        miss_data_MSE = torch.sum(((1-M_matrix) * global_normal - (1-M_matrix) * imputed_data)**2)
        self.log('val_MSE_loss', miss_data_MSE, on_epoch=True, prog_bar=True, logger=True)


    ## 优化器配置
    def configure_optimizers(self):
        logger.info('configure_optimizers 初始化开始...')
        # 选择优化器
        if self.args.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        
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


