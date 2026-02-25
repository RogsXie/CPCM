import torch
import torch.nn as nn
from mamba_ssm import Mamba
from thop import profile
class PatchEmbedding(nn.Module):

    def __init__(self,  in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1_1 = nn.Sequential(nn.Conv2d(self.in_channels[0], 32, 3, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=False))
        self.conv1_2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=False))
        self.conv2_1 = nn.Sequential(nn.Conv2d(self.in_channels[1], 32, 3, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=False))
        self.conv2_2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=False))
        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 128, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=False))

    def forward(self, x):
        x_h = self.conv1_2(self.conv1_1(x[0]))
        x_l = self.conv2_2(self.conv2_1(x[1]))

        x_out = torch.cat((x_h + x_l.mean(dim=1, keepdim=True).expand_as(x_h),
                                x_l + x_h.mean(dim=1, keepdim=True).expand_as(x_l)), dim=1)
        # x_out = torch.cat((x[0] ,x[1]) , dim=1)


        return self.conv3_1(x_out)

class ClusteringHead(nn.Module):
    def __init__(self, n_dim, n_class, alpha=1.):
        super(ClusteringHead, self).__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_class, n_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        pred_prob = self.get_cluster_prob(x)
        return pred_prob

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.cross_attn_spatial_to_spectral = nn.MultiheadAttention(d_model, num_heads=4,batch_first=True)
        self.cross_attn_spectral_to_spatial = nn.MultiheadAttention(d_model, num_heads=4,batch_first=True)
        self.norm_spatial = nn.LayerNorm(d_model)
        self.norm_spectral = nn.LayerNorm(d_model)

    def forward(self, spatial_feat, spectral_feat):
        attn_spatial, _ = self.cross_attn_spatial_to_spectral(
            query=spatial_feat,
            key=spectral_feat,
            value=spectral_feat
        )
        spatial_out = self.norm_spatial(spatial_feat + attn_spatial)

        attn_spectral, _ = self.cross_attn_spectral_to_spatial(
            query=spectral_feat,
            key=spatial_feat,
            value=spatial_feat
        )
        spectral_out = self.norm_spectral(spectral_feat + attn_spectral)

        return spatial_out+spectral_out

class CrossModalMamba(nn.Module):
    """Mamba Fusion Module with specific dimension processing"""

    def __init__(self, size=9, out_channels=128):
        super().__init__()

        self.seq_length =size*size

        self.cross_fusion = CrossAttentionFusion(out_channels)

        self.mamba_forward = Mamba(
            d_model=out_channels,
            d_state=32,
            d_conv=4,
            expand=2
        )

        self.mamba_spectral = Mamba(
            d_model=self.seq_length,
            d_state=32,
            d_conv=4,
            expand=2
        )
        self.norm_pre = nn.LayerNorm(out_channels)
        self.norm_post = nn.LayerNorm(out_channels)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.pool = nn.AdaptiveAvgPool1d(1)


    def _reshape_to_sequence(self, x):
        return x.flatten(2).permute(0, 2, 1)

    def _reshape_to_spatial(self, x):
        return x.permute(0, 2, 1)

    def _spectral_scan(self, x):
        x_spectral=x.permute(0, 2, 1)
        spectral_out = self.mamba_spectral(x_spectral)

        return spectral_out.permute(0, 2, 1)



    def forward(self, x):

        # seq1 = self._reshape_to_sequence(x)
        #
        # seq = self.norm_pre(seq1)
        #
        # mambaspa = self.mamba_forward(seq)
        #
        # mambaspe= self._spectral_scan(seq)
        #
        # mamba_out= self.cross_fusion(mambaspa, mambaspe)
        #
        # mamba_out = self.norm_post(mambaspe + seq)
        #
        # spatial_out = self._reshape_to_spatial(mamba_out)
        #
        # pooled = self.pool(spatial_out).squeeze(-1)
        # # pooled = self.pool(x).squeeze(-1).squeeze(-1)
        #
        # return pooled
        seq1 = self._reshape_to_sequence(x)

        seq = self.norm_pre(seq1)

        mambaspa = self.mamba_forward(seq)

        mambaspe= self._spectral_scan(seq)

        mamba_out= self.cross_fusion(mambaspa, mambaspe)

        mamba_out = self.norm_post(mamba_out + seq)
        # mamba_out = mamba_out + seq

        spatial_out = self._reshape_to_spatial(mamba_out)

        pooled = self.pool(spatial_out).squeeze(-1)

        return pooled

class Net(nn.Module):
    def __init__(self,  in_channels, n_class, dim_emebeding):
        super(Net, self).__init__()
        self.embedding_layer = PatchEmbedding( in_channels)

        self.mamba_path = CrossModalMamba()
        # self.mamba_path = CrossModalMamba()
        self.clustering_head = ClusteringHead(dim_emebeding, n_class, alpha=1) ## ContrastiveHead(512, 128)

    def forward(self, x_1, x_2):

        embedded_1 = self.embedding_layer(x_1)
        embedded_2 = self.embedding_layer(x_2)

        x_1 = self.mamba_path(embedded_1)
        x_2 = self.mamba_path(embedded_2)

        y_1 = self.clustering_head(x_1)
        y_2 = self.clustering_head(x_2)

        return y_1, y_2

    def forward_embedding(self, x):
        h = self.mamba_path(self.embedding_layer(x))
        return h

    def forward_cluster(self, x, return_h=False):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        h = self.mamba_path(self.embedding_layer(x))
        pred = self.clustering_head(h)
        labels = torch.argmax(pred, dim=1)
        if return_h:
            return labels, h
        return labels


from thop.vision.basic_hooks import count_linear


def count_mamba(mamba_module, x, y):
    """
    x: (input,)
    y: output

    Mamba 参数量和 FLOPs 估计公式：
    Params ≈ 3 * expand * d_model^2
    FLOPs ≈ B * L * 3 * expand * d_model^2
    """
    B, L, D = x[0].shape  # x shape: (B, L, D)
    expand = mamba_module.expand
    d_model = mamba_module.d_model

    params = 3 * expand * (d_model ** 2)
    flops = B * L * params

    mamba_module.total_ops += torch.DoubleTensor([flops])

from thop import profile, clever_format
from mamba_ssm import Mamba

custom_ops = {Mamba: count_mamba}
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model =Net((63,2), 6, 128).to(device)
#
#     hsi = torch.randn(1, 63, 9, 9).to(device)
#     lidar = torch.randn(1, 2, 9, 9).to(device)
#
#     pair = (hsi, lidar)
#     flops, params = profile(model, inputs=(pair, pair), custom_ops=custom_ops)
#
#     flops, params = clever_format([flops, params], "%.3f")
#
#     print("FLOPs:", flops)
#     print("Params:", params)
import time
import torch
import torch.nn as nn

try:
    from thop import profile
except Exception:
    profile = None

from mamba_ssm import Mamba

# ====== 你的网络代码（确保这些类已定义/或 import 进来）=====
# PatchEmbedding, ClusteringHead, CrossModalMamba, Net

# ---------------------------
# 1) thop 自定义 Mamba 统计
# ---------------------------
def count_mamba(mamba_module, x, y):
    """
    给 thop 用的自定义统计：
    将 Mamba 的主要计算近似为：
      MACs ≈ B * L * (3 * expand * d_model^2)

    注：这里把返回值视为 MACs（乘加次数）
    最终 FLOPs ≈ 2 * MACs（常用近似）
    """
    inp = x[0]                  # (B, L, D)
    B, L, D = inp.shape

    expand = getattr(mamba_module, "expand", 2)
    d_model = getattr(mamba_module, "d_model", D)

    macs_per_token = 3 * expand * (d_model ** 2)
    macs = B * L * macs_per_token

    mamba_module.total_ops += torch.DoubleTensor([macs])


custom_ops = {Mamba: count_mamba}


def build_inputs(device, batch_size=1, patch=9):
    """
    构造与你 Net.forward(x_1, x_2) 一致的输入：
    x_1, x_2 都是 (hsi, lidar) 的 tuple
    """
    hsi_1 = torch.randn(batch_size, 63, patch, patch, device=device)
    lid_1 = torch.randn(batch_size, 2,  patch, patch, device=device)
    hsi_2 = torch.randn(batch_size, 63, patch, patch, device=device)
    lid_2 = torch.randn(batch_size, 2,  patch, patch, device=device)

    x_1 = (hsi_1, lid_1)
    x_2 = (hsi_2, lid_2)
    return x_1, x_2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ===== 1) 实例化模型 =====
    # 你这里 dim_emebeding=128 是合理的：CrossModalMamba 输出 pooled (B, 128)
    model = Net(in_channels=(63, 2), n_class=6, dim_emebeding=128).to(device)
    model.eval()

    # ===== 2) 构造输入 =====
    x_1, x_2 = build_inputs(device=device, batch_size=1, patch=9)

    # ===== 3) 统计参数量 =====
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # ===== 4) 前向运行（验证能跑通）=====
    with torch.no_grad():
        y1, y2 = model(x_1, x_2)
    print(f"输出形状: y1={tuple(y1.shape)}, y2={tuple(y2.shape)}")
    print(f"输出示例: y1[0]={y1[0].detach().cpu()}")

    # ===== 5) thop 计算 MACs / FLOPs / Params =====
    if profile is not None:
        macs, params = profile(model, inputs=(x_1, x_2), custom_ops=custom_ops, verbose=False)

        print(f"计算量 MACs : {macs/1e6:.4f} M")
        print(f"计算量 FLOPs: {2*macs/1e6:.4f} M   (1 MAC ≈ 2 FLOPs)")
        print(f"参数量 Params: {params/1e3:.4f} K  (thop)")
    else:
        print("未安装 thop，跳过 MACs/FLOPs 统计。可执行: pip install thop")

    # ===== 可选：latency + peak memory（审稿人要求的 timing protocol）=====
    if device.type == "cuda":
        # warm-up
        with torch.no_grad():
            for _ in range(20):
                _ = model(x_1, x_2)
        torch.cuda.synchronize()

        iters = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(x_1, x_2)
        torch.cuda.synchronize()
        end = time.time()

        latency_ms = (end - start) / iters * 1000
        print(f"Inference latency: {latency_ms:.3f} ms (batch=1, {iters} runs avg)")

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x_1, x_2)
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {peak_mem_mb:.2f} MB (forward only)")


if __name__ == "__main__":
    main()
