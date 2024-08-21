import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_cls import SegmentationDataset
from dice import DiceLoss
from unet import UNet
from torchvision import transforms
from torchvision.utils import save_image


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
            images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # [batch, height, width, channels]
            preds_np = preds.cpu().numpy().transpose(0, 2, 3, 1)    # [batch, height, width, 1]

            # 오버레이 적용 및 이미지 저장
            for i in range(images_np.shape[0]):
                image = images_np[i]
                pred = preds_np[i]

                if image.shape[2] == 1:  # 만약 흑백 이미지라면 채널을 맞추기 위해 흑백 이미지를 3채널로 변환
                    image = np.repeat(image, 3, axis=2)
                
                # 원본 이미지와 예측 마스크를 오버레이
                overlay = np.zeros_like(image)
                overlay[:, :, 0] = pred.squeeze() * 255  # 빨간색 채널에 예측된 마스크를 추가
                overlayed_image = cv2.addWeighted(image, 1, overlay, 0.5, 0)

                # 이미지 저장 (원본 이미지, 예측된 마스크, 오버레이된 이미지)
                overlayed_image = overlayed_image.transpose(2, 0, 1)  # [channels, height, width] 형태로 변환
                save_path = os.path.join(save_dir, f"test_image_{idx * test_loader.batch_size + i + 1}.png")
                save_image(torch.tensor(overlayed_image), save_path)

    test_loss = test_loss / len(test_loader.dataset)
    return test_loss




# 하이퍼파라미터 및 경로 설정
data_dir = './dataset'  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_input_dir = os.path.join(data_dir, "test/inputs")
test_label_dir = os.path.join(data_dir, "test/labels")
save_dir = './test_results'
batch_size = 4


# 테스트 데이터셋 로드

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


test_dataset = SegmentationDataset(image_dir=test_input_dir, mask_dir=test_label_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 및 손실 함수 설정
model = UNet().to(device)
model.load_state_dict(torch.load('C:/pnx_seg/checkpoint/unet_epoch_2.pth'))  # 학습된 모델 로드

# 테스트 및 이미지 저장
test_loss = test_model_and_save_images(model, test_loader, DiceLoss(), device, save_dir)
print(f"Test Loss (Dice): {test_loss:.4f}")
