import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset1 import FlatDataset, ValidDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
    
    def setup(self, stage=None):
        # self.train_set = FlatDataset(csv_file=self.args.train_data,
        #                              pro_type_file=self.args.pro_type_file, 
        #                              replace_dict_file=self.args.replace_dict_file
        #                             )
        self.train_set = ValidDataset(miss_file=self.args.train_data,
                                     complete_file=self.args.valid_data,
                                     pro_type_file=self.args.pro_type_file,
                                     replace_dict_file=self.args.replace_dict_file
                                    )
        self.valid_set = ValidDataset(miss_file=self.args.train_data,
                                     complete_file=self.args.valid_data,
                                     pro_type_file=self.args.pro_type_file,
                                     replace_dict_file=self.args.replace_dict_file
                                    )
    
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
