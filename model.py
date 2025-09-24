import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class StandardizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(StandardizeLayer, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class VGPNet(nn.Module):
    def __init__(
        self,
        imean=torch.tensor([0, 0, 0], dtype=torch.float64),
        istd=torch.tensor([1, 1, 1], dtype=torch.float64),
    ):
        super().__init__()

        self.ccffm_gate = nn.Parameter(torch.tensor(0.5))

        self.reg_dim = 0

        self.seq = torch.nn.Sequential(
            StandardizeLayer(imean, istd),
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.reg_dim += 64

        print("fisheye module")
        self.fisheye_encoder = models.resnet18(weights="IMAGENET1K_V1")
        # for name, param in self.fisheye_encoder.named_parameters():
        #     if "layer4" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        num_features_fisheye = self.fisheye_encoder.fc.in_features
        self.fisheye_encoder.fc = nn.Identity()
        self.fisheye_fc = nn.Sequential(
            nn.Linear(num_features_fisheye, 64),
        )
        self.reg_dim += 64

        self.wb_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.ccffm = CCFFM(channels=64, heads=4)

    def forward(self, x, img):
        x = self.seq(x)  # N_sample 64
        N_sample = x.shape[0]

        img_f = self.fisheye_fc(self.fisheye_encoder(img))  # 1 64
        img_f = img_f.expand(N_sample, -1)  # N_sample 64

        # 有环境图像和鱼眼图像，CCFFM交互
        fused_feat = self.ccffm(
            x.unsqueeze(-1).unsqueeze(-1),  # to B C H W, in this case: N_sample 64
            img_f.unsqueeze(-1).unsqueeze(-1),  # to B C H W, in this case: N_sample 64
        )
        fused_feat = fused_feat.view(N_sample, -1)  # N_sample 128
        fused_feat = torch.tanh(fused_feat)
        fused_feat = self.ccffm_gate * fused_feat + (1 - self.ccffm_gate) * x
        feature = fused_feat

        x = self.wb_decoder(feature)

        weight = torch.sigmoid(x[:, 0])
        weight = torch.clamp(weight, min=0, max=1)
        bias = F.leaky_relu(x[:, 1])
        return weight, bias


class MHSA(nn.Module):
    """Multi-Head Self-Attention, 接收 q,k,v"""

    def __init__(self, channels, heads=4):
        super(MHSA, self).__init__()
        self.channels = channels
        self.heads = heads
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, q, k, v):
        # q,k,v: [B, N, C]
        B, N, C = q.size()
        head_dim = C // self.heads

        def reshape(x):
            return x.view(B, N, self.heads, head_dim).transpose(
                1, 2
            )  # [B, heads, N, head_dim]

        q, k, v = map(reshape, (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)  # [B, N, C]


class CrossChannelAttention(nn.Module):
    """双向 Cross-Attention"""

    def __init__(self, channels, heads=4):
        super(CrossChannelAttention, self).__init__()
        self.channels = channels
        self.heads = heads
        self.out_proj = nn.Linear(2 * channels, channels)

    def forward(self, q1, k1, v1, q2, k2, v2):
        B, N, C = q1.size()
        head_dim = C // self.heads

        def reshape(x):
            return x.view(B, N, self.heads, head_dim).transpose(1, 2)

        q1, k1, v1 = map(reshape, (q1, k1, v1))
        q2, k2, v2 = map(reshape, (q2, k2, v2))

        # x1 attend to x2
        scores12 = torch.matmul(q1, k2.transpose(-2, -1)) / (head_dim**0.5)
        attn12 = F.softmax(scores12, dim=-1)
        out1 = torch.matmul(attn12, v2)

        # x2 attend to x1
        scores21 = torch.matmul(q2, k1.transpose(-2, -1)) / (head_dim**0.5)
        attn21 = F.softmax(scores21, dim=-1)
        out2 = torch.matmul(attn21, v1)

        out1 = out1.transpose(1, 2).contiguous().view(B, N, C)
        out2 = out2.transpose(1, 2).contiguous().view(B, N, C)

        fused = torch.cat([out1, out2], dim=-1)  # [B, N, 2C]
        return self.out_proj(fused)  # [B, N, C]


class CCFFM(nn.Module):
    """Cross-Channel Feature Fusion Module"""

    def __init__(self, channels, heads=4):
        super(CCFFM, self).__init__()
        self.channels = channels
        self.heads = heads
        self.qkv_proj = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(6)]
        )  # 2x(q,k,v)

        self.mhsa_gnss = MHSA(channels, heads)
        self.mhsa_fisheye = MHSA(channels, heads)
        self.ccff = CrossChannelAttention(channels, heads)

        self.out_proj = nn.Linear(3 * channels, channels)

    def forward(self, x1, x2):
        # 输入: x1, x2 [B, C, H, W]
        B, C, H, W = x1.size()

        def flatten(x):
            return x.flatten(2).transpose(1, 2)  # [B, N, C], N=H*W

        x1_flat, x2_flat = map(flatten, [x1, x2])

        # QKV projection
        q1, k1, v1 = [proj(x1_flat) for proj in self.qkv_proj[:3]]
        q2, k2, v2 = [proj(x2_flat) for proj in self.qkv_proj[3:]]

        # 单模态 self-attention
        y_sa1 = self.mhsa_gnss(q1, k1, v1)  # [B, N, C]
        y_sa2 = self.mhsa_fisheye(q2, k2, v2)  # [B, N, C]

        # 跨模态 cross-attention
        y_c = self.ccff(q1, k1, v1, q2, k2, v2)  # [B, N, C]

        # 拼接并输出
        out = self.out_proj(torch.cat([y_sa1, y_sa2, y_c], dim=-1))  # [B, N, C]

        # reshape 回 feature map
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


if __name__ == "__main__":
    model = VGPNet(
        imean=torch.tensor([0, 0, 0], dtype=torch.float64),
        istd=torch.tensor([1, 1, 1], dtype=torch.float64),
        bool_gnss=True,
        bool_fisheye=True,
        bool_ccffm=True,
    )
    model.double()
    model.train()
    model.cuda()
    # print(model)
    x = torch.randn(21, 3).double().cuda()
    img = torch.randn(1, 3, 224, 224).double().cuda()
    weight, bias = model(x, img)
    print("weight:", weight)
    print("bias:", bias)
