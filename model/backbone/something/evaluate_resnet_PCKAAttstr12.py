import torch
import torch.nn as nn


__all__ = ['ResNet', 'resnet50', 'resnet101']

class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        super(SpatialAttentionLayer, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", bias=True)
        self.act = activation
        self.att = nn.Sigmoid()

    def forward(self, x, rx):
        xrx = torch.cat((x, rx), 1)
        xatt = self.att(self.conv2(self.act(self.conv1(xrx))))
        return xatt

class SpattialAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        super(SpattialAttentionBlock, self).__init__()
        ic = in_channels
        oc = out_channels
        ks = kernel_size
        self.attfeat = SpatialAttentionLayer(ic, oc, ks, activation)
    def forward(self, x, rx, rx_bef=None):
        f = self.attfeat(x, rx)
        af = x * f
        if rx_bef is not None:
            aff = af + rx_bef
            return aff
        else:
            return af




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        self.PCKA = True
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.PCKA:
            self.att_channel = 128
            self.activation = nn.ReLU(inplace=True)
            self.att1, self.att2, self.att3 = (SpattialAttentionBlock(self.att_channel * 2, self.att_channel, 3, self.activation),
                                               SpattialAttentionBlock(self.att_channel * 2, self.att_channel, 3, self.activation),
                                               SpattialAttentionBlock(self.att_channel * 2, self.att_channel, 3, self.activation))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def apply_conv_bn_relu(self, x):
        return self.relu(self.bn1(self.conv1(x)))


    def base_forward(self, x, PCKA=False):
        if PCKA:
            x_l_aug, x_l = x

            x_l_aug = self.apply_conv_bn_relu(x_l_aug)
            x_l = [self.apply_conv_bn_relu(x_l[i]) for i in range(3)]

            f1_l = self.att1(x_l[0], x_l_aug)
            f2_l = self.att2(x_l[1], x_l_aug, f1_l)
            x = self.att3(x_l[2], x_l_aug, f2_l)



        else:

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            # print(x.shape)

        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pretrained_path = "pretrained/%s.pth" % arch
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
        # print('load pretrained')
    return model


def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)
