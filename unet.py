import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path




## 하이퍼파라미터
lr = 1e-3
batch_size = 4
num_epoch = 1
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

# Unet 구현
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 파란색 화살표 conv 3x3, batch-normalization, ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        ## Contracting path, encoder part
        # encoder part 1
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # 빨간색 화살표(Maxpool)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # encoder part 2
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        # 빨간색 화살표(Maxpool)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # encoder part 3
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        # 빨간색 화살표(Maxpool)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # encoder part 4
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        # 빨간색 화살표(Maxpool)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # encoder part 5
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)



        ## Expansive path, Decoder part
        # Decoder part 5
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # 초록 화살표(Up Convolution)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # Decoder part 4
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512) # encoder part에서 전달된 512 채널 1개를 추가
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # 초록 화살표(Up Convolution)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # Decoder part 3
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        
        # 초록 화살표(Up Convolution)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # Decoder part 2
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # 초록 화살표(Up Convolution)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # Decoder part 1
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # class에 대한 output을 만들어주기 위해 1x1 conv
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x): # x는 input image, nn.Module의 __call__은 forward 함수를 저절로 실행 
        enc1_1 = self.enc1_1(x) # nn.Sequential객체는 __call__ Mx를 가지고 있어서 함수처럼 호출가능
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        # print(pool3.size())
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x

from torch.utils.data import Dataset

# Custom Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("L")  # 흑백 이미지로 불러옴
        mask = Image.open(mask_path).convert("L")  # 흑백 이미지로 불러옴

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 인스턴스화 및 손실함수, 옵티마이저 설정
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits (마지막 레이어에서 sigmoid를 하지 않은 경우)
optimizer = optim.Adam(model.parameters(), lr=lr)
# 손실 기록 리스트
train_losses = []
val_losses = []

# 학습 및 검증 루프
for epoch in range(num_epoch):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)  # 학습 손실 기록

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)

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
plt.show()
