#Pre-Trained ConvNext model

import torch
import torch.nn as nn
import torchvision.models as models


class ConvNext():
    def __init__(self,num_classes):
        #pre_trained modeli yükle
        #models.convnext_tiny(pre_trained=True)
        self.ConvNext = models.convnext_tiny(weights="IMAGENET1K_V1",pretrained=True)
        self.num_classes=num_classes
        # ConvNext modelinin son katmanlarını değiştir
        self.num_ftrs = self.ConvNext.classifier[2].in_features  # Son fully connected katmanın giriş özellik sayısı
        

    def model(self):   
        
        # New fully connected layers
        new_fc_layers = nn.Sequential(
            nn.Linear(self.num_ftrs, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64,self.num_classes),
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


# conv=ConvNext(6)
# model=conv.model()


# print(model.classifier)

# for param in model.classifier.parameters():
# 	print(param)

