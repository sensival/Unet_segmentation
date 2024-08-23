import torch
from PIL import Image
import numpy as np

# 마스크 이미지 로드
mask = Image.open("C:/pnx_seg/dataset/train/labels/3_train_1_.png").convert("L")  # 흑백 이미지로 변환

# Mask 정규화
mask = np.array(mask) / 255.0  # 0과 1로 정규화
mask = Image.fromarray(mask)

mask_np = np.array(mask)

# 유일한 픽셀 값 확인
unique_values = np.unique(mask_np)
print(f"Unique pixel values in mask: {unique_values}")

# 하얀색 클래스 확인
# 일반적으로 1 또는 255일 것임.
foreground_value = unique_values[-1]  # 하얀색 클래스는 보통 가장 큰 값
print(f"Foreground (class) pixel value: {foreground_value}")
