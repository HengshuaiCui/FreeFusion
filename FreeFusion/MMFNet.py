import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from torch.nn import init, Sequential
import numpy as np
import kornia

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            # padding = kernel_size // 2 -1
            padding = kernel_size // 2
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
class DSCResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSCResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

    def forward(self, x):
        return self.main(x) + x
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=1):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=1):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_res=1):
        super(Encoder, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=1),
        ])
        self.down12 = DownSample(base_channel * 2, base_channel * 2)
        self.down23 = DownSample(base_channel * 4, base_channel * 4)
    def forward(self, x):
        en_outputs = list()
        en_1 = self.feat_extract[0](x)
        res1 = self.Encoder[0](en_1)
        en_outputs.append(res1)

        en_2 = self.feat_extract[1](res1)
        res2 = self.Encoder[1](en_2)
        x_12 = self.down12(res2)
        en_outputs.append(x_12)

        en_3 = self.feat_extract[2](x_12)
        res3 = self.Encoder[2](en_3)
        x_23 = self.down23(res3)
        en_outputs.append(x_23)

        return en_outputs

class Decoder(nn.Module):
    def __init__(self, num_res=1, is_seg = False, num_classes=6):
        super(Decoder, self).__init__()
        base_channel = 32
        self.is_seg = is_seg
        self.num_classes = num_classes
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1, transpose=True),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
        ])
        self.up32 = UpSample(base_channel * 2,base_channel * 2)
        self.up21 = UpSample(base_channel,base_channel)

        self.Skips = nn.ModuleList([
            Skip(base_channel * 7, base_channel * 1),
            Skip(base_channel * 7, base_channel * 2)

        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        self.classifier = nn.Conv2d(base_channel, self.num_classes, kernel_size=1)
    def forward(self, inputs):
        # ----------Skip Connection-----------#
        inputs_12 = F.interpolate(inputs[0], scale_factor=0.5)#32 160 160
        inputs_32 = F.interpolate(inputs[2], scale_factor=2)#128 160 160
        inputs_21 = F.interpolate(inputs[1], scale_factor=2)#64 320 320
        inputs_31 = F.interpolate(inputs_32, scale_factor=2)#128 320 320

        skip_2 = self.Skips[1](inputs_12, inputs[1], inputs_32)  # 1+2+4
        skip_1 = self.Skips[0](inputs[0], inputs_21, inputs_31)#1+2+4

        res1 = self.Decoder[0](inputs[2])  #4
        de_3 = self.feat_extract[0](res1) #2
        x_32 = self.up32(de_3)# 64 160 160

        cat_2 = torch.cat([x_32, skip_2], dim=1)# 64+32
        x_2 = self.Convs[0](cat_2)
        res2 = self.Decoder[1](x_2)
        de_2 = self.feat_extract[1](res2)
        x_21 = self.up21(de_2)

        cat_1 = torch.cat([x_21, skip_1], dim=1)
        x_1 = self.Convs[1](cat_1)
        res3 = self.Decoder[2](x_1)
        if self.is_seg:
            seg_head = self.classifier(res3)
            return seg_head
        else:
            de_1 = self.feat_extract[2](res3)
            return de_1

class FFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFBlock, self).__init__()

        self.conv_1_1 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*2, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channels*2, out_channels, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, stride=1, relu=True),
        )
        self.conv_1_2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, stride=1, relu=True),
        )
        self.channel_weights_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_weights_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_2 = nn.Sequential(
            BasicConv(in_channels * 2, out_channels, kernel_size=3, stride=1, relu=True),
        )


    def forward(self, en1, en2):
        cat_1_1 = torch.cat([en1, en2], dim=1)
        cat_1_2 = torch.cat([en1, en2], dim=1)
        shallow_conv_1_1 = self.conv_1_1(cat_1_1)
        shallow_conv_1_2 = self.conv_1_2(cat_1_2)

        channel_weights_1 = self.channel_weights_1(shallow_conv_1_1)
        channel_weights_2 = self.channel_weights_2(shallow_conv_1_2)
        x_1 = en1 * channel_weights_1
        x_2 = en2 * channel_weights_2
        cat_2 = torch.cat([x_1, x_2], dim=1)
        fus = self.conv_2(cat_2)

        return fus
class SFBlock(nn.Module):
    def __init__(self):
        super(SFBlock, self).__init__()
        channels = [32, 64, 128]
        self.body_1 = DSCResBlock(channels[0], channels[0])
        self.body_2 = DSCResBlock(channels[1], channels[1])
        self.body_3 = DSCResBlock(channels[2], channels[2])
    def forward(self, inp_feats):
        seg_fea_1 = self.body_1(inp_feats[0])
        seg_fea_2 = self.body_2(inp_feats[1])
        seg_fea_3 = self.body_3(inp_feats[2])
        return [seg_fea_1, seg_fea_2, seg_fea_3]


class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x*y
class FDBlock(nn.Module):
    """ Feature Decoupled block"""
    def __init__(self, in_channels = [32, 64, 128]):
        super(FDBlock, self).__init__()
        self.sa_0 = SALayer(in_channels[0], reduction=4, bias=False)
        self.sa_1 = SALayer(in_channels[1], reduction=4, bias=False)
        self.sa_2 = SALayer(in_channels[2], reduction=4, bias=False)

    def forward(self, fus_fea, modality_fea):#B C H W -> 6 128 80 80
        m_0 = self.sa_0(modality_fea[0])
        x_0 = fus_fea[0] - m_0

        m_1 = self.sa_1(modality_fea[1])
        x_1 = fus_fea[1] - m_1

        m_2 = self.sa_2(modality_fea[2])
        x_2 = fus_fea[2] - m_2

        return [x_0, x_1, x_2]

class Skip(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Skip, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, h):
        super(SelfAttention, self).__init__()
        assert in_channels % h == 0
        self.in_channels = in_channels
        self.head_channels = in_channels // h
        self.h = h
        # key, query, value projections for all heads
        self.que_proj = nn.Linear(in_channels, self.h * self.head_channels)  # query projection
        self.key_proj = nn.Linear(in_channels, self.h * self.head_channels)  # key projection
        self.val_proj = nn.Linear(in_channels, self.h * self.head_channels)  # value projection
        self.out_proj = nn.Linear(self.h * self.head_channels, in_channels)  # output projection

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, N = x.shape[:2]
        q = self.que_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)  # b n c->b n h c/h->b h n c/h
        k = self.key_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 3, 1)  # b c c/h n
        v = self.val_proj(x).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)  # b h n c/h
        att = torch.matmul(q,k) / np.sqrt(self.head_channels)  # b c n n
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.head_channels)  # b c n c/h -> b n h c/h->b n c
        out = self.out_proj(out)

        return out

class CrossTaskModule(nn.Module):
    def __init__(self, in_channels,high=32, width=32):
        super(CrossTaskModule, self).__init__()
        # avgpool
        self.in_channels = in_channels
        self.h = 8
        self.high = high
        self.width = width
        self.avgpool = nn.AdaptiveAvgPool2d((self.high, self.width))
        self.ln_input = nn.LayerNorm(in_channels)
        self.ln_output = nn.LayerNorm(in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        self.sa = SelfAttention(self.in_channels, self.h)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
    def forward(self, fus, seg):
        B, C, H, W = fus.shape
        # AvgPooling for reduce the dimension due to expensive computation
        fus_fea = self.avgpool(fus)  # 20 20
        seg_fea = self.avgpool(seg)  # 20 20
        fus_fea = self.bn1(fus_fea)
        seg_fea = self.bn2(seg_fea)
        fus_flat = fus_fea.view(B, C, -1)  # flatten the feature
        seg_flat = seg_fea.view(B, C, -1)  # flatten the feature

        mt_flat = torch.cat((fus_flat, seg_flat), dim=2)

        mt_flat = mt_flat.permute(0, 2, 1)
        mt_flat = self.sa(mt_flat)  #B 2n c
        mt_flat = self.ln_output(mt_flat)  # B 2n c

        mt_flat = mt_flat.view(B, 2, self.high, self.width, self.in_channels)
        mt = mt_flat.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        fus_out = mt[:, 0, :, :, :].contiguous().view(B, self.in_channels, self.high, self.width)
        seg_out = mt[:, 1, :, :, :].contiguous().view(B, self.in_channels, self.high, self.width)

        fus_res = self.mlp1(fus_out)
        seg_res = self.mlp2(seg_out)

        return fus_res, seg_res

class CrossAttention(nn.Module):

    def __init__(self, in_channels, h):
        super(CrossAttention, self).__init__()
        assert in_channels % h == 0
        self.in_channels = in_channels
        self.head_channels = in_channels // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(in_channels, self.h * self.head_channels)  # query projection
        self.key_proj = nn.Linear(in_channels, self.h * self.head_channels)  # key projection
        self.val_proj = nn.Linear(in_channels, self.h * self.head_channels)  # value projection
        self.out_proj = nn.Linear(self.h * self.head_channels, in_channels)  # output projection

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, ctf, st):
        B, N = st.shape[:2]
        st_q = self.que_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)  # b n h c/h->b h n c/h
        ctf_k = self.key_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 1, 3)   # b n h c/h->b h n c/h
        ctf_v = self.val_proj(ctf).view(B, N, self.h, self.head_channels).permute(0, 2, 3, 1)  # b n h c/h->b h c/h n
        # Self-Attention
        att = torch.matmul(ctf_k, ctf_v) / np.sqrt(self.head_channels)

        # get attention matrix
        att = torch.softmax(att, -1)
        # output
        query_res = torch.matmul(att, st_q).permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.head_channels)  #b h n c/h->b n h c/h-> b n c
        query_res = self.out_proj(query_res)

        return query_res

class CTQModule(nn.Module):
    def __init__(self, in_channels, high=32, width=32):
        super(CTQModule, self).__init__()
        self.in_channels = in_channels
        self.h = 8
        self.high = high
        self.width = width
        self.avgpool = nn.AdaptiveAvgPool2d((self.high, self.width))
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        self.ln_input_1 = nn.LayerNorm(self.in_channels)
        self.ln_input_2 = nn.LayerNorm(self.in_channels)
        self.ln_output = nn.LayerNorm(self.in_channels)
        self.cs = CrossAttention(self.in_channels, self.h)
    def forward(self, ctf, st):
        B, C, H, W = st.shape
        ctf_fea = ctf
        # AvgPooling for reduce the dimension due to expensive computation
        st_fea = self.avgpool(st)  # 20 20

        ctf_flat = ctf_fea.view(B, C, -1).permute(0, 2, 1)  # B N C
        st_flat = st_fea.view(B, C, -1).permute(0, 2, 1)  # B N C

        query_out = self.cs(ctf_flat, st_flat)
        query_out = self.ln_output(query_out)
        query_out = query_out.permute(0, 2, 1).contiguous().view(B, C, self.high, self.width)

        query_out = F.interpolate(query_out, size=([H, W]), mode='bilinear')

        query_res = st + query_out

        return query_res
class MMFNet(nn.Module):
    def __init__(self, num_classes):
        super(MMFNet, self).__init__()
        self.encoder_1 = Encoder()
        self.encoder_2 = Encoder()
        self.decoder = Decoder(is_seg = False)
        self.ffb_1 = FFBlock(in_channels=32, out_channels=32)
        self.ffb_2 = FFBlock(in_channels=64, out_channels=64)
        self.ffb_3 = FFBlock(in_channels=128, out_channels=128)
        self.sfb = SFBlock()
        self.ctf_1 = CrossTaskModule(in_channels=32, high=32, width=32)  # 320
        self.ctf_2 = CrossTaskModule(in_channels=64, high=16, width=16)  # 160
        self.ctf_3 = CrossTaskModule(in_channels=128, high=8, width=8)  # 80
        self.ctq_fus_1 = CTQModule(in_channels=32, high=32, width=32)
        self.ctq_seg_1 = CTQModule(in_channels=32, high=32, width=32)
        self.ctq_fus_2 = CTQModule(in_channels=64, high=16, width=16)
        self.ctq_seg_2 = CTQModule(in_channels=64, high=16, width=16)
        self.ctq_fus_3 = CTQModule(in_channels=128, high=8, width=8)
        self.ctq_seg_3 = CTQModule(in_channels=128, high=8, width=8)
        self.fdb_1 = FDBlock()
        self.fdb_2 = FDBlock()
        self.seghead = Decoder(is_seg=True, num_classes=num_classes)
    def forward(self, ir, rgb):
        #--------------Encoder for ir and rgb--------------#
        ir_feas = self.encoder_1(ir)
        rgb_feas = self.encoder_2(rgb)
        # ----------Shared Decoder for ir and rgb-----------#
        recon_ir = self.decoder(ir_feas)
        recon_rgb = self.decoder(rgb_feas)

        # -----multi-modality image feature fusion stage----#
        fus_fea_1 = self.ffb_1(ir_feas[0], rgb_feas[0])
        fus_fea_2 = self.ffb_2(ir_feas[1], rgb_feas[1])
        fus_fea_3 = self.ffb_3(ir_feas[2], rgb_feas[2])
        fus_feas = [fus_fea_1, fus_fea_2, fus_fea_3]
        # -------get seg feature----------------------------#
        seg_feas = self.sfb(fus_feas)
        # ------------Dynamic interaction stage---------------#
        ctf_fus_1, ctf_seg_1 = self.ctf_1(fus_feas[0], seg_feas[0])
        ctf_fus_2, ctf_seg_2 = self.ctf_2(fus_feas[1], seg_feas[1])
        ctf_fus_3, ctf_seg_3 = self.ctf_3(fus_feas[2], seg_feas[2])

        # ------------Cross-Task Query stage----------------#
        query_fus_1 = self.ctq_fus_1(ctf_fus_1, fus_feas[0])  # B C H W
        query_seg_1 = self.ctq_seg_1(ctf_seg_1, seg_feas[0])  # B C H W
        query_fus_2 = self.ctq_fus_2(ctf_fus_2, fus_feas[1])  # B C H W
        query_seg_2 = self.ctq_seg_2(ctf_seg_2, seg_feas[1])  # B C H W
        query_fus_3 = self.ctq_fus_3(ctf_fus_3, fus_feas[2])  # B C H W
        query_seg_3 = self.ctq_seg_3(ctf_seg_3, seg_feas[2])  # B C H W

        query_fus = [query_fus_1, query_fus_2, query_fus_3]
        query_seg = [query_seg_1, query_seg_2, query_seg_3]
        #------------feature decoupling stage--------------#
        # from fusion feature decouped ir feature
        dec_ir = self.fdb_1(query_fus, rgb_feas)
        # from fusion feature decouped rgb feature
        dec_rgb = self.fdb_2(query_fus, ir_feas)
        # -----------reconstruction decouped ir and rgb--------------#
        recon_dec_ir = self.decoder(dec_ir)
        recon_dec_rgb = self.decoder(dec_rgb)
        # -----------seg stage------------------------------#
        seg_res = self.seghead(query_seg)
        #-----------training use the following code-----#
        return recon_ir, recon_rgb, recon_dec_ir, recon_dec_rgb, seg_res
        #-----------testing use the following code-----#
        # fus = self.decoder(query_fus)
        # return _, _, _, _, _, fus


