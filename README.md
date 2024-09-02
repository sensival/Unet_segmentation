# Pneumothorax Unet Segmentation model

**`Sole contributer`**


<br>

</aside>

# About U-net


![x1](https://github.com/user-attachments/assets/8a6884ad-3f49-483a-b050-c87b2c6c1ab3)
U-Net 모델은 의료 영상에서 뛰어난 성능을 보여주고 있습니다. 이 모델은 이미지의 세부 정보와 전체적인 구조를 동시에 잘 잡아낼 수 있는 특징이 있어, 복잡한 의료 영상에서도 정확하게 분할 작업을 수행할 수 있습니다. 특히, 적은 양의 데이터로도 높은 성능을 낼 수 있어, 다양한 의료 분야에서 널리 사용되고 있습니다.

네트워크 아키텍처는Encoder(왼쪽)와 Decoder(오른쪽)로 구성되어 있습니다. Encoder part 는 전형적인 합성곱 신경망의 아키텍처를 따릅니다. 이 경로는 두 개의 3x3 convolution을 반복적으로 적용하고, 각 합성곱 뒤에는 ReLU function을 적용합니다. 또한, 2x2 max pooling(stride 2) 연산이 다운샘플링을 위해 사용됩니다. 각 다운샘플링 단계에서 특징 채널(feature channel)의 수가 두 배로 증가합니다.

Decoder part의 각 단계는 특징 맵의 업샘플링으로 시작되며, 이어서 2x2 up-convolution이 적용되어 특징 채널의 수를 절반으로 줄입니다. 그런 다음, Encoder part에서 대응되는 위치의 특징 맵과의 concatenation이 이루어지고, 두 개의 3x3 convolution과 각각의  ReLU function이 뒤따릅니다. 마지막 층에서는 1x1 convolution을 사용하여 각 64-feature vector를 클래스로 매핑합니다. 전체적으로 이 네트워크는 23개의 합성곱 층을 가지고 있습니다.

**Reference:** https://ar5iv.labs.arxiv.org/html/1505.04597

# Development Environment


이 프로젝트에서는 UNet 모델을 사용하여 Pneumothorax dataset에 대한 이미지 분할 학습을 수행했습니다. 아래는 개발 환경의 세부 사항입니다.

- **프레임워크**:
    - PyTorch
- **모델 아키텍처**:
    - UNet
- **도구 및 IDE**:
    - Visual Studio Code
    - Remote Explorer
- **데이터셋**:
    - [SIIM-ACR Pneumothorax Segmentation | Kaggle](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)
- **하드웨어**:
    - **CPU**: AMD Ryzen 9 5900X
    - **GPU**: NVIDIA RTX 3090
- **운영체제**:
    - Ubuntu 20.04.5 LTS
<br>
<br>

# Dataset

- Train dataset:  10,675
- Validation dataset: 1,072
- Test dataset: 300

<br>
<div style="display: flex; flex-direction: row;">
<img src="https://github.com/user-attachments/assets/81acf419-440e-4c4a-a9b7-a0eb9f20a3f3" alt="Mask Image 2" width="200" />
  <img src="https://github.com/user-attachments/assets/bd849f9c-a8d6-4985-b82b-ea7f455f37fc" alt="Input Image 1" width="200" />
  <img src="https://github.com/user-attachments/assets/d212d9be-2d40-43c4-b132-1c517c012408" alt="Input Image 2" width="200" />
  <img src="https://github.com/user-attachments/assets/3a49b49f-5dbb-4ce7-87e9-bb15847c564b" alt="Mask Image 1" width="200" />
</div>


<br>
<br>
<br>


# Code


### Unet code

**References :** [2. Segmentation 모델 구현 (feat. UNet) :: Time Traveler (tistory.com)](https://89douner.tistory.com/300)

```python
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
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512) 
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
        self.fc = nn.Conv2d(in_channels=64, out_channels=1,
					       
					        kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x) 
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

```

### Dice loss

**References :** [[딥러닝] Dice Coefficient 설명, pytorch 코드(segmentation 평가방법) (tistory.com)](https://minimin2.tistory.com/179)

```python
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice=(2.*intersection+smooth)/(inputs.sum()+targets.sum()+smooth)  
        
        return 1 - dice 
```

### Training code

**Developed with assistance from ChatGPT**

```python
## 하이퍼파라미터
lr = 1e-4
batch_size = 4
num_epoch = 200
data_dir = './dataset'  
origins_folder =os.path.join(data_dir, "train/inputs")
masks_folder = os.path.join(data_dir, "train/labels")
val_input_dir = os.path.join(data_dir, "val/inputs")
val_label_dir = os.path.join(data_dir, "val/inputs")
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

train_dataset = SegmentationDataset(image_dir=origins_folder,
							 mask_dir=masks_folder, transform=train_transform)
val_dataset = SegmentationDataset(image_dir=val_input_dir,
							 mask_dir=val_label_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, 
							shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
						 shuffle=False, num_workers=4)

# 모델 인스턴스화 및 손실함수, 옵티마이저 설정
model = UNet().to(device)
criterion_BCE = nn.BCEWithLogitsLoss()  
criterion_dice = DiceLoss()
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
        loss = criterion_BCE(outputs, masks)

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
            loss = criterion_BCE(outputs, mask)
            
            val_loss += loss.item() * images.size(0)
    
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)  # 검증 손실 기록
```

### Dataset preprocessing code

**Developed with assistance from ChatGPT**

```python
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
```

### Test Prediction code(Dice loss)

**Developed with assistance from ChatGPT**

```python
def test_model_and_save_images(model, test_loader, criterion, device, save_dir):
    model.eval()
    test_loss = 0.0
    os.makedirs(save_dir, exist_ok=True)  # 저장할 디렉토리 생성
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            # Sigmoid를 사용하여 0~1 사이로 변환
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()  # 0.5를 기준으로 이진화

            # CPU로 데이터 이동 및 numpy 변환
            images_np = images.cpu().numpy().transpose(0, 2, 3, 1) 
            preds_np = preds.cpu().numpy().transpose(0, 2, 3, 1)   

            # 오버레이 적용 및 이미지 저장
            for i in range(images_np.shape[0]):
                image = images_np[i]
                pred = preds_np[i]

                if image.shape[2] == 1:  
                    image = np.repeat(image, 3, axis=2)
                
                # 원본 이미지와 예측 마스크를 오버레이
                overlay = np.zeros_like(image)
                overlay[:, :, 0] = pred.squeeze() * 255 
                overlayed_image = cv2.addWeighted(image, 1, overlay, 0.5, 0)

                # 이미지 저장 (원본 이미지, 예측된 마스크, 오버레이된 이미지)
                overlayed_image = overlayed_image.transpose(2, 0, 1) 
                save_path = os.path.join(save_dir, 
									          f"test_image_{idx*test_loader.batch_size+i+1}.png")
                save_image(torch.tensor(overlayed_image), save_path)

    test_loss = test_loss / len(test_loader.dataset)
    return test_loss
```

# Result


## Train/validation loss

- 이진 교차 엔트로피 손실(Binary Cross Entropy Loss)를 계산하는 PyTorch 함수인 nn.BCEWithLogitsLoss()를 사용했습니다.
- 이 함수는 모델의 출력(logits)과 실제 레이블(0 또는 1)을 비교하여 손실을 계산합니다.

![image](https://github.com/user-attachments/assets/0a37bafe-6d08-4194-a77a-cacecdd5e825)

- **σ(x)**: 시그모이드 함수
- **x**: 모델의 출력(logits)
- y: 실제 레이블 (0 또는 1)
![image 1](https://github.com/user-attachments/assets/700bba01-34a3-408d-b0be-f3d64833db4f)

## Test loss

Segmentation 모델이 예측한 Test dataset의 mask image와 레이블된 mask image의 비교를 통해 평가했습니다.

- **평가 항목**
1. **Dice loss**
    
    Dice 계수는 두 이진 집합 간의 겹침을 측정하며, **0에서 1** 사이의 값을 가집니다.
    

![%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C_(4)](https://github.com/user-attachments/assets/bef29ddd-37e9-475e-adc4-b5912f84be69)

여기서 X는 예측된 픽셀 집합, Y는 실제 라벨 집합이며, 이 식에서의 ∣X∩Y∣는 예측값과 실제값 간의 교집합을 의미합니다.

![image 2](https://github.com/user-attachments/assets/e8e2872b-a6bd-4349-90f3-2cb93a60cd03)

  Dice Score는 IoU Score보다 True positive 영역에 더 가중치를 두는 평가 방식이라서 병변이 작은 Pneumothroax 모델을 보다 합리적으로 평가할 수 있습니다.

  Dice Loss는 다음과 같이 계산된다. 
![image 3](https://github.com/user-attachments/assets/c2f629f6-92be-40d1-8071-eb156f7761ee

*** smooth는 분모가 0이 되는 것을 방지하기 위한 값* 

```python
intersection = (inputs * targets).sum()                            
dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
```
<br>
<br>

2. **Recall**
    
    Recall(재현율) 은 모델이 실제로 양성인 픽셀 중에서 모델이 올바르게 예측한 픽셀의 비율을 나타내며,  **0에서 1** 사이의 값을 가집니다.
    
![images_(1)](https://github.com/user-attachments/assets/a2c18000-e414-49ea-b34d-ed885d7cd93c)

Recall은 다음과 같이 정의 됩니다.
![image 4](https://github.com/user-attachments/assets/d859868c-2f43-46d8-916e-14fa898374da)
![image 5](https://github.com/user-attachments/assets/4b51781e-4044-4f61-89db-7f375b7b410f)

*** smooth는 분모가 0이 되는 것을 방지하기 위한 값* 

```python
intersection = (inputs * targets).sum()     
recall = (intersection + smooth) / (targets.sum() + smooth) 
```

<br>
<br>
<br>


- Test 결과

|  | 20 epochs | 50 epochs | 200 epochs |
| --- | --- | --- | --- |
| Diceloss | 0.8944 | 0.7348 | 0.5865 |
| Recall | 0.5748 | 0.6379 | 0.6264 |
<br>
<br>

## Test prediction
![test_image_6](https://github.com/user-attachments/assets/367a17ac-9e42-43dd-be7e-8e76ef9ac165)
![test_image_13](https://github.com/user-attachments/assets/d9e9c946-5b78-4b2f-a8d5-7224e0cdecde)
![test_image_74](https://github.com/user-attachments/assets/8f908f7b-c5f0-4d89-92c4-de1277d8f7f4)
![test_image_80](https://github.com/user-attachments/assets/fb44b673-c67f-4d76-9052-c4f0bebb9542)
![test_image_155](https://github.com/user-attachments/assets/3dd5c007-04b2-468a-9c27-8edacb20900b)
![test_image_248](https://github.com/user-attachments/assets/bb23e0e0-ade5-4775-b38a-ea75f6454c6e)

<br>
<br>

# 후기


YOLO 모델 파인튜닝 경험을 바탕으로, 하이퍼파라미터 조정과 레이어 구성을 직접 제어하는 코드를 작성해 보고자 PyTorch를 활용하여 U-Net 기반의 세그멘테이션 모델을 학습했습니다. 초기에는 GPU에서 학습을 위해 CUDA와 CUDNN을 설치하려 했으나, 버전 호환 문제로 인해 어려움을 겪었습니다. 이에, CUDA가 사용 가능한 서버에 Remote Explorer를 통해 접속하여, U-Net GitHub 리포지토리에서 제공하는 컨테이너 환경을 활용해 학습을 성공적으로 진행할 수 있었습니다.

Validation loss가 안정되지 않는 문제를 해결하지 못한 점은 아쉬웠습니다. 이는 Validation 데이터셋의 질적인 문제일 가능성이 있으며, 폐렴이나 장폐색과 같이 병변의 대비가 명확한 질환과 달리 Pneumothorax는 판단하기 어려운 사례가 많아 데이터셋의 질을 평가하기 어려웠습니다. 이러한 이유로 Test loss도 높게 나타났습니다.

이를 개선하기 위해서는 병변이 더 명확하게 드러나는 양질의 데이터셋이 필요할 것으로 보입니다. 특히, 병변이 작은 경우가 많고, 마스크에 검정 이미지(정상 케이스)가 다수 포함된 데이터셋보다는, 병변이 큰 유질환 데이터들이 더욱 중요할 것입니다.

의료 영상 데이터에서는 유질환을 정확히 걸러내는 것이 중요하다고 판단하여, Dice 지표를 변형하여 Recall 수치로도 성능을 평가해 보았습니다. 다음 프로젝트에서는 성능 평가를 위한 다양한 지표들을 찾아보고 적용해볼 계획입니다.