import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hswish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., self.inplace) / 6.

class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., self.inplace) / 6.

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, in_chs, out_chs, ksize, stride, activation=nn.ReLU):
        super().__init__()
        padding = (ksize - 1) // 2
        # 深度卷积
        self.dw_conv = nn.Conv2d(in_chs, in_chs, ksize, stride, padding, groups=in_chs, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_chs)
        self.dw_act = activation(inplace=True)
        
        # 逐点卷积
        self.pw_conv = nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_chs)
        self.pw_act = activation(inplace=True)

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw_conv(x)))
        x = self.pw_act(self.pw_bn(self.pw_conv(x)))
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super().__init__()
        reduced_chs = int(in_chs * se_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.act2 = hsigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.act1(self.reduce(scale))
        scale = self.act2(self.expand(scale))
        return x * scale

# 坐标卷积层
class CoordConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, padding=0, bias=True):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_chs + 2, out_chs, kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        # 获取输入的尺寸 (batch_size, C, H, W) 和设备信息
        batch_size, _, h, w = x.size()
        device = x.device
        
        # 创建坐标矩阵，并确保它们在相同的设备上
        h_channel = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(batch_size, -1, -1, w).to(device)
        w_channel = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(batch_size, -1, h, -1).to(device)
        
        # 将坐标矩阵添加到输入张量中
        x_coord = torch.cat([x, h_channel, w_channel], dim=1)
        return self.conv(x_coord)


class MobileNetV3Block(nn.Module):
    """标准MobileNetV3瓶颈块"""
    def __init__(self, in_chs, out_chs, kernel_size, 
                 stride, exp_ratio=4, se_ratio=0.25,
                 act_layer=nn.ReLU):
        super().__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.use_res = stride == 1 and in_chs == out_chs
        self.has_se = se_ratio is not None and se_ratio > 0.

        # 扩展层
        self.conv_exp = nn.Conv2d(in_chs, mid_chs, 1, bias=False)
        self.bn_exp = nn.BatchNorm2d(mid_chs)
        self.act_exp = act_layer(inplace=True)

        # 深度可分离卷积
        self.dw_conv = nn.Conv2d(
            mid_chs, mid_chs, kernel_size,
            stride, (kernel_size-1)//2, 
            groups=mid_chs, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(mid_chs)
        self.act_dw = act_layer(inplace=True)

        # SE模块
        if self.has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio)

        # 输出层
        self.conv_proj = nn.Conv2d(mid_chs, out_chs, 1, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        residual = x
        
        # 扩展阶段
        x = self.act_exp(self.bn_exp(self.conv_exp(x)))
        
        # 深度卷积
        x = self.act_dw(self.bn_dw(self.dw_conv(x)))
        
        # SE模块
        if self.has_se:
            x = self.se(x)
        
        # 投影层
        x = self.bn_proj(self.conv_proj(x))
        
        # 残差连接
        if self.use_res:
            x += residual
            
        return x

# 修改了初始层和瓶颈配置，加入了CoordConv
class MobileNetV3Lite(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        # 初始层使用CoordConv替代普通Conv
        self.conv_stem = CoordConv(in_channels, 8, 3, stride=2, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(8)
        self.act_stem = hswish()

        # Bottleneck配置调整
        cfg = [
            (8,  16,  4, 3, 2, None,    nn.ReLU),   # 提高初始通道数，提升细节捕捉能力
            (16,  16,  3, 3, 1, None,    nn.ReLU),
            (16,  24,  3, 5, 2, 0.25,  nn.ReLU),
            (24,  24,  3, 5, 1, 0.25,  nn.ReLU),
            (24,  40,  3, 5, 2, 0.25,  nn.ReLU),
            (40, 40,  3, 5, 1, 0.25,  nn.ReLU),
            (40, 40,  3, 5, 1, 0.25,  nn.ReLU),
        ]

        self.blocks = nn.Sequential()
        for i, (in_chs, out_chs, exp_ratio, k, s, se_ratio, act) in enumerate(cfg):
            self.blocks.add_module(
                f"bottleneck{i}",
                MobileNetV3Block(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    exp_ratio=exp_ratio,
                    kernel_size=k,
                    stride=s,
                    se_ratio=se_ratio,
                    act_layer=act
                )
            )

        self.conv_head = CoordConv(40, 120, 1, bias=False)
        self.bn_head = nn.BatchNorm2d(120)
        self.act_head = hswish()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(120, 64),
            hswish(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.act_stem(self.bn_stem(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act_head(self.bn_head(self.conv_head(x)))
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x