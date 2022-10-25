import pytorch_lightning as pl
import torch
from torch import nn

from .model1 import VAE
from .model2 import VAE2
from .model3 import VAE3
from .model4 import VAE4
from .model5 import VAE5
import pickle
from loss_function import vae_loss_2


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
        if self.args.model_type == 'model1':
            self.model = VAE(dim=self.args.dim, nhead=self.args.nhead)
        elif self.args.model_type == 'model2':
            self.model = VAE2(dim=self.args.dim, nhead=self.args.nhead)
        elif self.args.model_type == 'model3':
            self.model = VAE3(dim=self.args.dim, nhead=self.args.nhead)
        elif self.args.model_type == 'model4':
            self.model = VAE4(dim=self.args.dim)
        elif self.args.model_type == 'model5':
            pro_types = pickle.load(open(self.args.pro_type_file, 'rb'))
            self.model = VAE5(dim=self.args.dim, pro_types=pro_types)

        ## 参数初始化
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def training_step(self, batch, batch_idx):
        src_data, miss_data, M_matrix = batch['src_data'], batch['miss_data'], batch['miss_matrix']
        global_max, global_min = batch['global_max'], batch['global_min']
        normal_data = batch['normal_data']
        imputed_data, mu, log_var = self.model(miss_data, M_matrix) # [batch, dim]
        # imputed_data = imputed_data * (global_max-global_min) + global_min # 恢复原来的值

        loss, MSE_loss, kl_div = vae_loss_2(normal_data, imputed_data, M_matrix, mu, log_var)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('kl_div', kl_div, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('MSE_loss', MSE_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_data, miss_data, M_matrix = batch['src_data'], batch['miss_data'], batch['miss_matrix']
        global_max, global_min = batch['global_max'], batch['global_min']
        normal_data = batch['normal_data']
        imputed_data, mu, log_var = self.model(miss_data, M_matrix)

        # imputed_data = imputed_data * global_max + global_min # 恢复原来的值
        loss, MSE_loss, _ = vae_loss_2(normal_data, imputed_data, M_matrix, mu, log_var)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_MSE_loss', MSE_loss, on_epoch=True, prog_bar=True, logger=True)


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
                                                                   T_max=200,
                                                                   eta_min=1e-6,
                                                                   verbose=True,
                                                                   last_epoch=-1)
            logger.info('configure_optimizers 初始化结束...')
            return [optimizer], [scheduler]

        elif self.args.lr_scheduler == 'None':
            logger.info('configure_optimizers 初始化结束...')
            return optimizer


