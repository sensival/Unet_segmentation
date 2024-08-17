import os
import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from PIL import Image

import matplotlib.pyplot as plt





## 하이퍼파라미터
lr = 1e-3
batch_size = 4
num_epoch = 1
data_dir = './dataset/images'  
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


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # List files in directories
        input_dir = os.path.join(self.data_dir, 'inputs')
        label_dir = os.path.join(self.data_dir, 'labels')
        
        self.lst_input = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
        self.lst_label = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
        
        # Ensure both lists are of the same length and corresponding
        assert len(self.lst_input) == len(self.lst_label), "Number of input and label images do not match"

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        input_filename = self.lst_input[index]
        label_filename = self.lst_label[index]
        
        input_path = os.path.join(self.data_dir, 'inputs', input_filename)
        label_path = os.path.join(self.data_dir, 'labels', label_filename)
        
        # Load images using PIL
        input_image = Image.open(input_path).convert('RGB')
        label_image = Image.open(label_path).convert('RGB')
        
        # Convert images to numpy arrays
        input_array = np.array(input_image, dtype=np.float32)
        label_array = np.array(label_image, dtype=np.float32)
        
        # Normalize images to [0, 1]
        input_array /= 255.0
        label_array /= 255.0
        
        # Convert to grayscale if the images are essentially grayscale but in RGB format
        if input_array.ndim == 3 and input_array.shape[2] == 3:
            input_array = np.mean(input_array, axis=2, keepdims=True)
        if label_array.ndim == 3 and label_array.shape[2] == 3:
            label_array = np.mean(label_array, axis=2, keepdims=True)
        
        # Add channel dimension
        input_array = np.transpose(input_array, (2, 0, 1))  # Convert to (C, H, W)
        label_array = np.transpose(label_array, (2, 0, 1))  # Convert to (C, H, W)
        
        data = {'input': input_array, 'label': label_array}

        if self.transform:
            data = self.transform(data)
        
        return data
    
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        # Ensure data is in numpy format
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if isinstance(input, torch.Tensor):
            input = input.numpy()
        
        # Convert numpy arrays to PyTorch tensors
        label = torch.from_numpy(label).float()
        input = torch.from_numpy(input).float()
        
        data = {'label': label, 'input': input}
        
        return data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform 적용해서 데이터 셋 불러오기
transform = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), ToTensor()])
dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'),transform=transform)

# 불러온 데이터셋, 배치 size줘서 DataLoader 해주기
loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)

# val set도 동일하게 진행
dataset_val = Dataset(data_dir=os.path.join(data_dir,'val'),transform = transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size , shuffle=True)

# 네트워크 불러오기
net = UNet().to(device) # device : cpu or gpu

# loss 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr = lr ) 

# 기타 variables 설정
num_train = len(dataset_train)
num_val = len(dataset_val)

num_train_for_epoch = np.ceil(num_train/batch_size) # np.ceil : 소수점 반올림
num_val_for_epoch = np.ceil(num_val/batch_size)

# 기타 function 설정
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1) # device 위에 올라간 텐서를 detach 한 뒤 numpy로 변환
fn_denorm = lambda x, mean, std : (x * std) + mean 
fn_classifier = lambda x :  1.0 * (x > 0.5)  # threshold 0.5 기준으로 indicator function으로 classifier 구현

# Tensorbord
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val = SummaryWriter(log_dir = os.path.join(log_dir,'val'))
# 네트워크 저장하기
# train을 마친 네트워크 저장 
# net : 네트워크 파라미터, optim  두개를 dict 형태로 저장
def save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},'%s/model_epoch%d.pth'%(ckpt_dir,epoch))

# 네트워크 불러오기
def load(ckpt_dir,net,optim):
    if not os.path.exists(ckpt_dir): # 저장된 네트워크가 없다면 인풋을 그대로 반환
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir) # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
    ckpt_lst.sort(key = lambda f : int(''.join(filter(str,isdigit,f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optim,epoch


# 네트워크 학습시키기
start_epoch = 0
net, optim, start_epoch = load(ckpt_dir = ckpt_dir, net = net, optim = optim) # 저장된 네트워크 불러오기

for epoch in range(start_epoch+1,num_epoch +1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train,1): # 1은 뭐니 > index start point
        # forward
        label = data['label'].to(device)   # 데이터 device로 올리기     
        inputs = data['input'].to(device)
        output = net(inputs) 

        # backward
        optim.zero_grad()  # gradient 초기화
        loss = fn_loss(output, label)  # output과 label 사이의 loss 계산
        loss.backward() # gradient backpropagation
        optim.step() # backpropa 된 gradient를 이용해서 각 layer의 parameters update

        # save loss
        loss_arr += [loss.item()]

        # tensorbord에 결과값들 저정하기
        label = fn_tonumpy(label)
        inputs = fn_tonumpy(fn_denorm(inputs,0.5,0.5))
        output = fn_tonumpy(fn_classifier(output))

        writer_train.add_image('label', label, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('input', inputs, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    
    # validation
    with torch.no_grad(): # validation 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
        net.eval() # 네트워크를 evaluation 용으로 선언
        loss_arr = []

        for batch, data in enumerate(loader_val,1):
            # forward
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)

            # loss 
            loss = fn_loss(output,label)
            loss_arr += [loss.item()]
            print('valid : epoch %04d / %04d | Batch %04d \ %04d | Loss %04d'%(epoch,num_epoch,batch,num_val_for_epoch,np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_classifier(output))

            writer_val.add_image('label', label, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('input', inputs, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # epoch이 끝날때 마다 네트워크 저장
        save(ckpt_dir=ckpt_dir, net = net, optim = optim, epoch = epoch)

writer_train.close()
writer_val.close()