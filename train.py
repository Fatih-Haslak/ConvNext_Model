import os
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data.dataloader import Dataloader
from model.convnext import ConvNext
#from utils.logger import SimpleProgressBar
#from pytorch_lightning.plugins import DDPPlugin
#from lightning.pytorch.loggers import TensorBoardLogger
#from lightning.pytorch.loggers import WandbLogger


class ConvNextModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int=6,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        model_conv=ConvNext(num_classes) # model
     
        self.convnext=model_conv.model()
        self.criterion = nn.CrossEntropyLoss() # loss function
        #self.optimizer_model= self.optimizers()

    def forward(self, z):
        return self.convnext(z) #torch.image generator forward

    def training_step(self, batch,batch_idx):
        #image, #label
        img, label = batch
        self.optimizer_model= self.optimizers()
        self.predicted_label = self(img) #img
        
        #calculate loss
        loss = self.criterion(self.predicted_label, label)
        
        self.log("train_Loss", loss, prog_bar=True) #loggla
        
        self.manual_backward(loss,retain_graph=True)# backward
        self.optimizer_model.step()
        self.optimizer_model.zero_grad()
       
    def validation_step(self, batch, batch_idx):
        x, y = batch
        predict = self.convnext(x)
        loss = self.criterion(predict,y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,sync_dist=True)
       
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_model = torch.optim.Adam(self.convnext.parameters(), lr=lr, betas=(b1, b2))
    
        return optimizer_model

    def on_train_epoch_end(self):
        print('\n') #for logger
    def on_validation_epoch_end(self):
        print('\n') #for logger

if __name__ == "__main__": 
    data_path="/home/fatih/Desktop/convnext/seg_train"
    batch_size=16
    data = Dataloader(data_path,batch_size)
    
    model=ConvNextModel()    
    train,val=data.load_data() #load data

    trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=50,
    strategy="ddp",
    #precision="64-true",
    #callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)]
    #plugins=DDPPlugin(find_unused_parameters=False),
    #strategy="ddp_find_unused_parameters_true",
    #num_nodes=1,
    )

    for param in model.parameters():
        print(param.dtype)

    
    trainer.fit(model, train, val)
    trainer.save_checkpoint("best_model.ckpt")
        
