#Pre-Trained ConvNext model

import torch
import torch.nn as nn
import torchvision.models as models
#from torchcontrib.optim import SWA

class ConvNext():
    def __init__(self,num_classes):
        #load pre_trained model
        #models.convnext_base(pre_trained=True)
        self.ConvNext = models.convnext_base(weights="IMAGENET1K_V1",pretrained=True)
        self.num_classes=num_classes
        #Change last linear layer block
        self.num_ftrs = self.ConvNext.classifier[2].in_features  # Son fully connected katmanın giriş özellik sayısı
        print(self.ConvNext.classifier[2])

    def model(self):   
        
        # New fully connected layers
        new_fc_layers = nn.Sequential(
            nn.Linear(self.num_ftrs, 1000),
            nn.ReLU(),
            nn.LayerNorm(1000),
            nn.Linear(1000,self.num_classes),
            #nn.Softmax(dim=1)  
        ) 

        self.ConvNext.classifier[2]=new_fc_layers #
        

        # Modelin parametrelerini dondurma /Pre-Trained modellerde kullanılır)
        for param in self.ConvNext.parameters():
            param.requires_grad = False

        # Yeni eklenen katmanın parametrelerini güncelleme / Modelin son katmanını egıtıcez
        for param in self.ConvNext.classifier.parameters():
            param.requires_grad = True
         
        

        new_model=self.ConvNext

        return new_model

conv=ConvNext(6)
model=conv.model()
print("New",model.classifier[2])

# print(model.classifier)

# for param in model.classifier.parameters():
# 	print(param)

