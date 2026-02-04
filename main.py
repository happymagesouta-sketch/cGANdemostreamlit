import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils

# =====================
# ハイパーパラメータ
# =====================
latent_dim = 10
n_classes = 10
img_size = 28

# =====================
# デバイス設定
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Generator 定義
# =====================
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm
