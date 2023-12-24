import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
import pytorch_lightning as pl
from model.convnext import ConvNext
import numpy as np
import cv2
import glob
import os.path as osp
import torch.nn.functional as F

def test(test_img_folder:str,model):
    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        
        # Görüntüyü oku ve boyutlandır
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        img = img * 1.0 / 255
        #güncelle1
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)
        # Modeli kullanarak tahmin yap
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            
            #probabilities =  output.numpy()
            #predicted_class = output.argmax(output[0])
        # Çıktıyı yeniden boyutlandır ve kaydet
        #output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        #output = (output * 255.0).round()
        print(output)
        print(output.argmax())

if __name__ == "__main__":
    #model checkpoint
    checkpoint = "/home/fatih/Desktop/convnext/best_model.ckpt"
    # Test görüntülerinin klasörü
    test_img_folder = "/home/fatih/Desktop/convnext/seg_train/buildings/*"
    #gpu 
    device = torch.device(0)
    #load model
    num_classes=6
    model = ConvNext(num_classes).model()
    checkpoint = torch.load(checkpoint)
    
    model.load_state_dict(checkpoint, strict=False)
    # Diğer model ayarları
    model = model.to(device)
    model.eval()
    test(test_img_folder,model)
