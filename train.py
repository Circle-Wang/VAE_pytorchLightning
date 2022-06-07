from dataModule import DInterface
from modelModule import MInterface
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning import Trainer
import logging
import hydra #hydra-core
from omegaconf import DictConfig


logger = logging.getLogger("train")

def main(hparams):
    '''
    hparams: 读取config.yaml得到的对象。
    '''
    model = MInterface(hparams)
    dataloader = DInterface(hparams)
    
    tfname = f'{hparams.dataset_name} lr {hparams.lr} lr_scheduler {hparams.lr_scheduler}'.replace(
        ' ', '_').replace('.', '_') # 储存模型相关参数以及tensorboard的路径
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
                    gpus=eval(hparams.gpus), \
                    precision=hparams.precision , \
                    max_epochs=hparams.max_epochs, \
                    # accelerator='dp', \
                    logger=tb_logger, \
                    callbacks=[checkpoint_callback, progressBar_callback, lr_monitor])

    if hparams.checkpoint_path:
        logger.info(f'Start training from {hparams.checkpoint_path}')
        trainer.fit(model, dataloader, ckpt_path=hparams.checkpoint_path)
    else:
        logger.info(f'Start training...')
        trainer.fit(model, dataloader)
    
    
@hydra.main(config_path='', config_name="config") # 读取当前当前工作环境中的config
def my_app(args: DictConfig):
    main(args)

if __name__ == '__main__':
    my_app()