import torch
import torch.nn as nn
import numpy as np
from models.unet_parts import *

# Unet backbone
class unetback(nn.Module):
    def __init__(self):
        super(unetback, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65

        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # def forward(self, x):
        """Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          x_hw: image shape
          x4: extracted features
        """
        x_hw = x.shape[2:]

        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        return x4, x_hw

# interest point detector
class detHead(nn.Module):
    def __init__(self):
        super(detHead, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65

        self.relu = torch.nn.ReLU(inplace=True)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)

    def forward(self, x4, x_hw=0):
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))

        return semi

# descriptor head
class descHead(nn.Module):
    def __init__(self):
        super(descHead, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65

        self.relu = torch.nn.ReLU(inplace=True)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)

    def forward(self, x4, x_hw=0):
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        return desc

# sematic head
class semHead(nn.Module):
    def __init__(self, n_classes=133):
        super(semHead, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65

        self.relu = torch.nn.ReLU(inplace=True)

        # Seg Head.
        self.convDS = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnS1 = nn.BatchNorm2d(c5)
        self.convSout = torch.nn.Conv2d(c5, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x4, x_hw):
        sem = self.convDS(x4)
        sem = self.bnS1(sem)
        sem = self.relu(sem)
        sem = self.convSout(sem)
        sem = F.interpolate(sem, x_hw, mode="bilinear", align_corners=False)

        return sem

# model dict for central dir method
def get_senner_model(config, device, semantic):
    model = {}

    model["enc"] = unetback()
    model["semi"] = detHead()
    model["desc"] = descHead()
    if semantic:
        model["sem"] = semHead()
        model["sem"].to(device)

    model["enc"].to(device)
    model["semi"].to(device)
    model["desc"].to(device)

    return model
