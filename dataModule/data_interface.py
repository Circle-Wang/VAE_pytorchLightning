import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset1 import FlatDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
    
    def setup(self, stage=None):
        self.train_set = FlatDataset(self.args.train_data,
                                    missing_ratio=self.args.missing_ratio,
                                    data_norm=self.args.data_norm)
        self.valid_set = FlatDataset(self.args.val_data,
                                    missing_ratio=self.args.missing_ratio,
                                    is_test=True,
                                    data_norm=self.args.data_norm)
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_set, 
                             batch_size = self.batch_size, 
                             collate_fn = self.train_set.collater, 
                             shuffle = True,
                             num_workers = self.args.num_workers)
        return train_dataloader
    
    def val_dataloader(self):
        valid_dataloader = DataLoader(self.valid_set, 
                             batch_size=self.batch_size, 
                             collate_fn=self.valid_set.collater, 
                             shuffle=False,
                             num_workers=self.args.num_workers)
        return valid_dataloader
