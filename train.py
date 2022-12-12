from dataModule import DInterface
from modelModule import MInterface
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning import Trainer
import logging
# import hydra #hydra-core
from omegaconf import DictConfig, OmegaConf
import torch
import argparse



logger = logging.getLogger("train")

def main(hparams):
    '''
    hparams: 读取config.yaml得到的对象。
    '''
    model = MInterface(hparams)
    dataloader = DInterface(hparams)
    
    tfname = f'{hparams.dataset_name} {hparams.model_type}'.replace(' ', '_').replace('.', '_') # 储存模型相关参数以及tensorboard的路径
    
    # 设置回调函数
    lr_monitor = LearningRateMonitor() # 记录学习率回调函数
    progressBar_callback = TQDMProgressBar(refresh_rate=20)   # 进度条回调函数
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor='val_loss', mode='min', save_last=True) # 监控保存val_loss用于保存模型参数
    
    # 设置tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(hparams.save_dir,
                                             version=hparams.version, 
                                             name=tfname) # 会在file_path路径下创建一个name的文件夹，并且在该文件下创建一个version文件夹来储存
    # 训练设置
    trainer = Trainer(gradient_clip_val=0.5, \
                    gpus=hparams.gpus, \
                    precision=hparams.precision , \
                    max_epochs=hparams.max_epochs, \
                    # accelerator='dp', \
                    logger=tb_logger, \
                    callbacks=[checkpoint_callback, progressBar_callback, lr_monitor])

    if hparams.checkpoint_path:
        logger.info(f'Start training from {hparams.checkpoint_path}')
        # trainer.fit(model, dataloader, ckpt_path=hparams.checkpoint_path)
        # 以下方式相当于只载入参数,无其他信息
        checkpoint = torch.load(hparams.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, dataloader)
    else:
        logger.info(f'Start training...')
        trainer.fit(model, dataloader)
    
    
# @hydra.main(config_path='', config_name="config_Breast") # 读取当前当前工作环境中的config
def my_app(args: DictConfig):
    main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型训练配置文件名")
    parser.add_argument('--config', default=None, help = "模型训练配置文件名")
    args = parser.parse_args()
    
    if args.config is not None:
        hparams = OmegaConf.load(args.config) # 读取配置文件
        my_app(hparams)
    else:
        print("请输入正确配置文件名")