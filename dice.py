# dice 계산
import torch.nn.functional as F
import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs) # sigmoid를 통과한 출력이면 주석처리
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice 


class DiceLoss_for_test(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss_for_test, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        dice_per_class = []

        if inputs.shape[1] == 1:  # 단일 클래스인 경우
            #input_class = F.sigmoid(inputs).view(-1)
            target_class = targets.view(-1)

            intersection = (input_class * target_class).sum()
            dice = (2. * intersection + smooth) / (input_class.sum() + target_class.sum() + smooth)
            dice_per_class.append(dice.item())

        else:  # 다중 클래스인 경우
            for c in range(self.num_classes):
                input_class = inputs[:, c, :, :]
                target_class = targets[:, c, :, :]

                input_class = F.sigmoid(input_class).view(-1)
                target_class = target_class.view(-1)

                intersection = (input_class * target_class).sum()
                dice = (2. * intersection + smooth) / (input_class.sum() + target_class.sum() + smooth)
                dice_per_class.append(dice.item())

        return 1 - torch.mean(torch.tensor(dice_per_class)), dice_per_class

