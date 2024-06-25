from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from torch.nn import LayerNorm, Softmax
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch import nn
from timm.models.layers import DropPath


class DepthwiseConv1dxDx1WithDilation(nn.Module):
    def __init__(self, input_channels, kernel_size, dilation=1):
        super(DepthwiseConv1dxDx1WithDilation, self).__init__()
        if dilation == 1:
            # 1xd的深度卷积核
            self.depthwise_conv_1xd = nn.Conv2d(input_channels, input_channels, kernel_size=(1, kernel_size),
                                                padding=(0, kernel_size // 2), groups=input_channels,
                                                dilation=(1, dilation))
            # dx1的深度卷积核
            self.depthwise_conv_dx1 = nn.Conv2d(input_channels, input_channels, kernel_size=(kernel_size, 1),
                                                padding=(kernel_size // 2, 0), groups=input_channels,
                                                dilation=(dilation, 1))
        else:
            # 1xd的深度卷积核
            self.depthwise_conv_1xd = nn.Conv2d(input_channels, input_channels, kernel_size=(1, kernel_size),
                                                padding=(0, dilation), groups=input_channels,
                                                dilation=(1, dilation))
            # dx1的深度卷积核
            self.depthwise_conv_dx1 = nn.Conv2d(input_channels, input_channels, kernel_size=(kernel_size, 1),
                                                padding=(dilation, 0), groups=input_channels,
                                                dilation=(dilation, 1))

    def forward(self, x):

        x = self.depthwise_conv_1xd(x)
        x = self.depthwise_conv_dx1(x)
        return x


class ConvBN(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation=1):
        super(ConvBN, self).__init__()
        self.conv = DepthwiseConv1dxDx1WithDilation(in_channels, kernel_size, dilation)
        # self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3)
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # 通道维度放到最后
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # 通道维度放到最前
        x1 = self.relu(x)
        x = self.pool(x1)
        return x, x1


class ConvLayerWithNormalization(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayerWithNormalization, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.ln = nn.LayerNorm(in_channels)  # 层归一化
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # 通道维度放到最后
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class LocalSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(LocalSpatialAttentionModule, self).__init__()

        # Fully Connected Layer for Fc
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU()
        )
        # Convolutional Layer for C1×1
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Sigmoid activation for AttL
        self.sigmoid = nn.Sigmoid()
        self.dilated_conv2 = ConvBN(in_channels, kernel_size=3, dilation=1)
        self.dilated_conv3 = ConvBN(in_channels, kernel_size=3, dilation=2)
        self.dilated_conv4 = ConvBN(in_channels, kernel_size=3, dilation=3)

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm = LayerNorm(in_channels, eps=1e-6)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = x + conv1
        conv2 = self.conv2(conv1)
        conv2 = conv1 + conv2
        conv3 = self.conv3(conv2)
        conv3 = conv2 + conv3
        conv_output = self.conv(conv3)
        # Sigmoid activation for AttL
        att_weights = self.sigmoid(conv_output)  # AttL(F2_i)

        # Lsa = AttL(F2_i) * F2_i + F2_i
        lsa_output = att_weights * x + x
        lsa_output1 = lsa_output.permute(0, 2, 3, 1)
        lsa_output1 = self.norm(lsa_output1)
        lsa_output1 = lsa_output1.permute(0, 3, 1, 2)
        dilated2 = self.dilated_conv2(lsa_output1)
        dilated2 = F.gelu(dilated2)
        dilated3 = self.dilated_conv3(lsa_output1)
        dilated3 = F.gelu(dilated3)
        dilated4 = self.dilated_conv4(lsa_output1)
        dilated4 = F.gelu(dilated4)
        output = dilated2 + dilated3 + dilated4
        output = self.conv1x1(output)
        output = F.gelu(output)
        out = output + lsa_output1
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, drop_rate=0.):
        super(FeatureExtractor, self).__init__()
        self.lsa = LocalSpatialAttentionModule(in_channels)
        self.gam = GAM_Module(in_channels)
        self.cam = CAM_Module(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x1 = self.cam(x)
        lsa = self.lsa(x1)
        gam = self.gam(x1)
        output = lsa + gam
        output = self.conv1x1(output)
        output = x + self.drop_path(output)
        return output


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class GAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(GAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=3, padding=1,
                                    groups=in_dim // 8)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=3, padding=1,
                                  groups=in_dim // 8)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        # self.dwc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 3), padding=1, groups=in_dim)
        self.dilated_conv2 = ConvBN(in_dim, kernel_size=3, dilation=1)
        self.dilated_conv3 = ConvBN(in_dim, kernel_size=3, dilation=2)
        self.dilated_conv4 = ConvBN(in_dim, kernel_size=3, dilation=3)

        self.conv1x1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.norm = LayerNorm(in_dim, eps=1e-6)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        features = self.gamma * out + x
        features1 = features.permute(0, 2, 3, 1)
        features1 = self.norm(features1)
        features1 = features1.permute(0, 3, 1, 2)
        dilated2 = self.dilated_conv2(features1)
        dilated2 = F.gelu(dilated2)
        dilated3 = self.dilated_conv3(features1)
        dilated3 = F.gelu(dilated3)
        dilated4 = self.dilated_conv4(features1)
        dilated4 = F.gelu(dilated4)
        output = dilated2 + dilated3 + dilated4
        output = self.conv1x1(output)
        output = F.gelu(output)
        out = output + features
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        # out = out + x
        return out


class DSUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(DSUNet, self).__init__()
        filters1 = [32, 72, 144, 256, 320]
        self.stemk = Stem(in_channels=3, out_channels=filters1[1])
        self.down1 = ConvLayerWithNormalization(filters1[1], filters1[2], kernel_size=3, stride=1, padding=1)
        self.down2 = ConvLayerWithNormalization(filters1[2], filters1[3], kernel_size=3, stride=1, padding=1)
        self.down3 = ConvLayerWithNormalization(filters1[3], filters1[4], kernel_size=3, stride=1, padding=1)
        self.FeatureExtractor1 = FeatureExtractor(filters1[1])
        self.FeatureExtractor2 = FeatureExtractor(filters1[2])
        self.FeatureExtractor3 = FeatureExtractor(filters1[3])
        self.FeatureExtractor4 = FeatureExtractor(filters1[4])

        # swintransformer
        swins3 = swin_tiny_patch4_window7_224(pretrained=True)
        self.patch_embeds3 = swins3.patch_embed
        self.layer1 = swins3.layers[0].blocks
        self.downs2 = swins3.layers[1].downsample
        self.layer2 = swins3.layers[1].blocks
        self.downs3 = swins3.layers[2].downsample
        self.layer3 = swins3.layers[2].blocks
        self.downs4 = swins3.layers[3].downsample
        self.layer4 = swins3.layers[3].blocks

        filters = [72, 144, 256, 320]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        out_channels = 32
        self.final_conv1 = nn.ConvTranspose2d(filters[0], out_channels, 4, 2, padding=1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.final_conv2 = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.final_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.final_relu3 = nn.ReLU(inplace=True)
        self.final_conv4 = nn.Conv2d(out_channels, n_classes, 3, padding=1)

        filters2 = [96, 192, 384, 768]
        self.sdecoder4 = DecoderBottleneckLayer(filters2[3], filters2[2])
        self.sdecoder3 = DecoderBottleneckLayer(filters2[2], filters2[1])
        self.sdecoder2 = DecoderBottleneckLayer(filters2[1], filters2[0])
        # out_channels = 32
        self.sfinal_conv1 = nn.ConvTranspose2d(filters2[0], out_channels, 4, 2, padding=1)
        self.sfinal_relu1 = nn.ReLU(inplace=True)
        self.sbn1 = nn.BatchNorm2d(out_channels)
        self.sfinal_conv2 = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, padding=1)
        self.sfinal_relu2 = nn.ReLU(inplace=True)
        self.sbn2 = nn.BatchNorm2d(out_channels)
        self.sfinal_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.sfinal_relu3 = nn.ReLU(inplace=True)
        self.sfinal_conv4 = nn.Conv2d(out_channels, n_classes, 3, padding=1)

    def forward(self, x):
        s1 = self.patch_embeds3(x)
        s1 = self.layer1(s1)
        s2 = self.downs2(s1)
        s2 = self.layer2(s2)
        s3 = self.downs3(s2)
        s3 = self.layer3(s3)
        s4 = self.downs4(s3)
        s4 = self.layer4(s4)
        s1 = s1.permute(0, 3, 1, 2)
        s2 = s2.permute(0, 3, 1, 2)
        s3 = s3.permute(0, 3, 1, 2)
        s4 = s4.permute(0, 3, 1, 2)
        ds4 = self.sdecoder4(s4) + s3
        ds3 = self.sdecoder3(ds4) + s2
        ds2 = self.sdecoder2(ds3) + s1
        ds1 = self.sfinal_conv1(ds2)
        ds1 = self.sbn1(ds1)
        ds1 = self.sfinal_relu1(ds1)
        ds1 = self.sfinal_conv2(ds1)
        ds1 = self.sbn2(ds1)
        ds1 = self.sfinal_relu2(ds1)
        ds1 = self.sfinal_conv3(ds1)
        ds1 = self.sfinal_relu3(ds1)
        ds1 = self.sfinal_conv4(ds1)
        ds1 = torch.sigmoid(ds1)
        sout = (ds1 > 0.5).float()
        x1 = sout * x

        k1, k0 = self.stemk(x1)
        k1 = self.FeatureExtractor1(k1)
        k2 = self.down1(k1)
        k2 = self.FeatureExtractor2(k2)
        k3 = self.down2(k2)
        k3 = self.FeatureExtractor3(k3)
        k4 = self.down3(k3)
        k4 = self.FeatureExtractor4(k4)
        d4 = self.decoder4(k4) + k3
        d3 = self.decoder3(d4) + k2
        d2 = self.decoder2(d3) + k1
        out1 = self.final_conv1(d2)
        out1 = self.bn1(out1)
        out1 = self.final_relu1(out1)

        out2 = self.final_conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.final_relu2(out2)

        out = self.final_conv3(out2)
        out = self.final_relu3(out)

        out = self.final_conv4(out)
        out = torch.sigmoid(out)
        return {"out": out, "outs": ds1}


if __name__ == '__main__':
    x = torch.randint(0, 255, [2, 3, 224, 224]).type(torch.FloatTensor)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    de = DSUNet()
    print(de)
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(de, x)
    param = sum(p.numel() for p in de.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(de, x)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

    print(flop_count_table(flops, max_depth=1))
    out = de(x)
    print(out['out'].shape)
    print(out['outs'].shape)
