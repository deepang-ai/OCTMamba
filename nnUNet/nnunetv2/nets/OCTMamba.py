import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import get_act_layer
from mamba_ssm import Mamba


class FeatureExtractor(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        super().__init__()

        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            padding=1,
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class ConvNorm(nn.Sequential):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
    ):
        super().__init__()
        self.add_module(
            "c",
            nn.Conv3d(in_dim, out_dim, ks, stride, pad, dilation, groups, bias=False),
        )
        self.add_module("bn", nn.BatchNorm3d(out_dim))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv3d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class SelfAdaptiveFeatureExtraction(nn.Module):
    def __init__(self, ed, kernel_size):
        super().__init__()
        self.conv = ConvNorm(ed, ed, kernel_size, 1, (kernel_size - 1) // 2, groups=ed)
        self.conv1 = ConvNorm(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
            [1, 1, 1, 1],
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class ChannelFusionEnhancement(nn.Module):
    def __init__(self, in_dim, hidden_dim, act_layer):
        super().__init__()
        self.conv1 = ConvNorm(in_dim, hidden_dim, 1, 1, 0)
        self.act = get_act_layer(act_layer)
        self.conv2 = ConvNorm(hidden_dim, in_dim, 1, 1, 0, bn_weight_init=0)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class SelectiveStateSpace(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv,  
            expand=expand,  
        )

    def forward(self, x):
        B, C, W, H, D = x.shape
        if not x.is_contiguous():
            x = x.contiguous()
        
        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)
        
        x = self.norm(x)
        x = self.mamba(x)
        
        x = x.permute(0, 2, 1)
        x = x.view(B, C, W, H, D)
        assert x.size() == torch.Size((B, C, W, H, D))
        return x


class OCTMambaBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, mlp_ratio, kernel_size, act_layer, stride=4, num_heads=8
    ):
        super(OCTMambaBlock, self).__init__()

        self.conv = nn.Conv3d(in_dim, out_dim, 3, 1, 1)
        self.norm = nn.GroupNorm(out_dim // 2, out_dim)
        self.act = get_act_layer(act_layer)
        
        self.sfe = SelfAdaptiveFeatureExtraction(out_dim, kernel_size)
        self.sss = SelectiveStateSpace(dim=out_dim)
        self.cfe = ChannelFusionEnhancement(out_dim, out_dim * mlp_ratio, act_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        identity = x 
        
        x = self.sfe(x) 
        
        x = self.sss(x)
        
        x = self.cfe(x)
        
        return identity + x


class OCTMamba(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        num_heads=(2, 4, 8, 16),
        strides=(4, 2, 2, 1),
        coord=True,
        dropout=0.3,
    ):
        super(OCTMamba, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._coord = coord
        
        self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.upsampling = nn.Upsample(scale_factor=2)
        
        self.feature_extractor = FeatureExtractor(
            3,
            in_channels,
            channels[0],
            "gelu",
            ("instance", {"affine": True}),
            True,
            dropout,
        )
        
        # Encoder Stages
        self.encoder1 = OCTMambaBlock(
            channels[0],
            channels[1],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[0],
            stride=strides[0],
        )

        self.encoder2 = OCTMambaBlock(
            channels[1],
            channels[2],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[1],
            stride=strides[1],
        )

        self.encoder3 = OCTMambaBlock(
            channels[2],
            channels[3],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[2],
            stride=strides[2],
        )

        self.encoder4 = OCTMambaBlock(
            channels[3],
            channels[4],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[3],
            stride=strides[3],
        )
        
        # Decoder Stages
        self.decoder1 = OCTMambaBlock(
            channels[3] + channels[4],
            channels[3],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[3],
            stride=strides[3],
        )

        self.decoder2 = OCTMambaBlock(
            channels[2] + channels[3],
            channels[2],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[2],
            stride=strides[2],
        )

        self.decoder3 = OCTMambaBlock(
            channels[1] + channels[2],
            channels[1],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[1],
            stride=strides[1],
        )

        if self._coord:
            num_channel_coord = 3
        else:
            num_channel_coord = 0
            
        self.decoder4 = OCTMambaBlock(
            channels[0] + channels[1] + num_channel_coord,
            channels[0],
            2,
            kernel_size=3,
            act_layer="gelu",
            num_heads=num_heads[0],
            stride=strides[0],
        )

        # Segmentation Head
        self.seg_head = nn.Conv3d(channels[0], self._out_channels, 1, 1, 0)

    def forward(self, input, coordmap=None):
        # Feature Extractor
        x_feat = self.feature_extractor(input)
        x = self.pooling(x_feat)

        # Encoder
        x_enc1 = self.encoder1(x)
        x = self.pooling(x_enc1)

        x_enc2 = self.encoder2(x)
        x = self.pooling(x_enc2)

        x_enc3 = self.encoder3(x)
        x = self.pooling(x_enc3)

        x_enc4 = self.encoder4(x) 

        # Decoder
        x = self.upsampling(x_enc4)
        x = torch.cat([x, x_enc3], dim=1)
        x_dec1 = self.decoder1(x)

        x = self.upsampling(x_dec1)
        x = torch.cat([x, x_enc2], dim=1)
        x_dec2 = self.decoder2(x)

        x = self.upsampling(x_dec2)
        x = torch.cat([x, x_enc1], dim=1)
        x_dec3 = self.decoder3(x)

        x = self.upsampling(x_dec3)

        if self._coord and (coordmap is not None):
            x = torch.cat([x, x_feat, coordmap], dim=1)
        else:
            x = torch.cat([x, x_feat], dim=1)

        x_dec4 = self.decoder4(x)


        output = self.seg_head(x_dec4)
        return output