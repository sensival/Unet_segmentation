# 메인코드
import os
import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from pathlib import Path

# custom class
from dataset_cls import SegmentationDataset
from dice import DiceLoss
from unet import UNet



## 하이퍼파라미터
lr = 1e-4
batch_size = 4
num_epoch = 200
data_dir = './dataset'  
origins_folder =os.path.join(data_dir, "train/inputs")
masks_folder = os.path.join(data_dir, "train/labels")
val_input_dir = os.path.join(data_dir, "val/inputs")
val_label_dir = os.path.join(data_dir, "val/inputs")

models_folder = Path("models")
images_folder = Path("images")

ckpt_dir = './checkpoint' # 트레이닝된 데이터 저장
log_dir = './log' # 텐서보드 로그
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 데이터셋 불러오기
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = SegmentationDataset(image_dir=origins_folder, mask_dir=masks_folder, transform=train_transform)
val_dataset = SegmentationDataset(image_dir=val_input_dir, mask_dir=val_label_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 인스턴스화 및 손실함수, 옵티마이저 설정
model = UNet().to(device)
criterion_BCE = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits (마지막 레이어에서 sigmoid를 하지 않은 경우)
criterion_dice = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# 손실 기록 리스트
train_losses = []
val_losses = []


# 학습 및 검증 루프
num=0
for epoch in range(num_epoch):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion_dice(outputs, masks)
        #print('tr output:',outputs.min(), outputs.max()) 
        #print('tr mask:',masks.min(), masks.max()) 

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        #print('running {}'.format(num))
        #num = num+1
    print('tr loss:',train_loss,'\n','tr image size:', images.size(0), '\n','tr len(train_loader):',len(train_loader.dataset), '\n')
    train_loss = train_loss / len(train_loader.dataset) 
    train_losses.append(train_loss)  # 학습 손실 기록

    # Validation step
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion_dice(outputs, masks)
            #print('val output:', outputs.min(), outputs.max()) 
            #print('val mask:', masks.min(), masks.max()) 
            
            val_loss += loss.item() * images.size(0)
    print('va loss:',val_loss,'\n','va image size:', images.size(0), '\n','va len(val_loader):',len(val_loader.dataset), '\n')
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)  # 검증 손실 기록

    print(f"Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 모델 저장
    if epoch % 10 == 0 or epoch == num_epoch - 1:
        torch.save(model.state_dict(), os.path.join(ckpt_dir,  f"unet_epoch_{epoch+1}.pth"))

# 학습 과정 시각화
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epoch+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epoch+1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Val Loss')
plt.legend()
plt.grid(True)
plt.savefig('train_val_loss_plot.png')
plt.show()


