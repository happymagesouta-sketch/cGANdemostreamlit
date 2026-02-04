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
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# =====================
# モデルロード
# =====================
@st.cache_resource
def load_generator():
    # 参照コードに基づき weights_only=False を明示
    model = torch.load(
        "GANgenerator.pth",
        map_location=device,
        weights_only=False
    )
    model.eval()
    return model

generator = load_generator()

# =====================
# Streamlit UI
# =====================
st.title("Conditional GAN Image Generator (MNIST)")
st.write(f"Using device: **{device}**")

target_label = st.selectbox(
    "生成したい数字ラベル",
    list(range(n_classes))
)

num_images = st.slider(
    "生成する画像枚数",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("画像生成"):
    labels = torch.full(
        (num_images,),
        target_label,
        dtype=torch.long,
        device=device
    )
    z = torch.randn(num_images, latent_dim, device=device)

    with torch.no_grad():
        gen_imgs = generator(z, labels)

    # [-1, 1] → [0, 1]
    gen_imgs = (gen_imgs + 1) / 2

    grid = vutils.make_grid(gen_imgs, nrow=num_images, normalize=False)
    st.image(
        grid.permute(1, 2, 0).cpu().numpy(),
        caption=f"Label {target_label}"
    )
