import torch
import torch.nn as nn


import numpy as np
import matplotlib.pyplot as plt
import os
__all__ = ['ResNet', 'resnet50', 'resnet101']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
    """2 Levels deformable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, act_type="leaky"):
        super(SpattialAttentionBlock, self).__init__()
        ic = in_channels
        oc = out_channels
        ks = kernel_size

        self.attfeat = SpatialAttentionLayer(ic, oc, ks, act_type)

    def forward(self, x, rx):
        """
        x (Tensor): features
        rx (Tensor): reference features
        """
        f = self.attfeat(x, rx)
        af = x * f

        return af


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
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     norm_layer(64),
        #     nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     norm_layer(64),
        #     nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     norm_layer(128),
        #     nn.ReLU(inplace=True)
        # )





        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        # self.att_channel = 128
        # self.activation = nn.LeakyReLU()
        # self.att1 = SpattialAttentionBlock(self.att_channel * 2, self.att_channel, 3, self.activation)
        # self.att2 = SpattialAttentionBlock(self.att_channel * 2, self.att_channel, 3, self.activation)

        self.att_channel = 128
        self.activation = nn.LeakyReLU()

        # self.att_blocks = [SpattialAttentionBlock(self.att_channel * 2, self.att_channel, 3, self.activation) for _ in range(12)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.att_blocks = [SpattialAttentionBlock(self.att_channel * 2, self.att_channel, 3, self.activation).to(self.device) for _ in range(12)]

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

    # def visualize_feature_maps_heatmap(self, feature_maps, num_slices=5):
    #     # 选择要可视化的特征图（例如，第一个特征图）
    #     feature_map = feature_maps[0].cpu().detach().numpy()
    #
    #     # 选择一些切片进行可视化
    #     slice_indices = np.linspace(0, feature_map.shape[2] - 1, num_slices, dtype=int)
    #
    #     fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    #
    #     for i, idx in enumerate(slice_indices):
    #         im = axes[i].imshow(feature_map[:, :, idx], cmap='hot')
    #         axes[i].set_title(f'Slice {idx}')
    #         axes[i].axis('off')
    #         fig.colorbar(im, ax=axes[i])
    #
    #     plt.show()


    # 假设你的特征图存储在变量feature_map中
    # feature_map的形状为(2, 256, 160, 160)




    # def base_forward(self, x, guide, val, id):
    #     if guide:
    #         x0, x1 = x
    #
    #         x_fea_lst_0 = []
    #         for ii in range(len(x0)):
    #             # print(ii)
    #             x_0_tmp = self.relu(self.bn1(self.conv1(x0[ii])))
    #             x_fea_lst_0.append(x_0_tmp)
    #         x0 = torch.stack(x_fea_lst_0)
    #         x0 = torch.sum(x0, dim=0)
    #
    #         x0 = [att_block(x_i, x0) for att_block, x_i in zip(self.att_blocks, x_fea_lst_0)]
    #         x0 = torch.sum(torch.stack(x0), dim=0)
    #
    #         x_fea_lst_1 = []
    #         for ii in range(len(x1)):
    #             # print(ii)
    #             x_1_tmp = self.relu(self.bn1(self.conv1(x1[ii])))
    #             x_fea_lst_1.append(x_1_tmp)
    #         x1 = torch.stack(x_fea_lst_1)
    #         x1 = torch.sum(x1, dim=0)
    #
    #         x1 = [att_block(x_i, x1) for att_block, x_i in zip(self.att_blocks, x_fea_lst_1)]
    #         x1 = torch.sum(torch.stack(x1), dim=0)
    #
    #         x = torch.cat((x0, x1))
    #
    #     elif val:
    #         x_fea_lst = []
    #         for ii in range(len(x)):
    #             # print(ii)
    #             x_tmp = self.relu(self.bn1(self.conv1(x[ii])))
    #             x_fea_lst.append(x_tmp)
    #         x = torch.stack(x_fea_lst)
    #         x = torch.sum(x, dim=0)
    #
    #         x = [att_block(x_i, x) for att_block, x_i in zip(self.att_blocks, x_fea_lst)]
    #         x = torch.sum(torch.stack(x), dim=0)
    #
    #         # 创建保存图像的文件夹
    #         output_dir = 'heatmaps'
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #
    #         # 遍历每个batch
    #         for batch_index in range(x.shape[0]):
    #             # 获取当前batch的特征图
    #             current_batch = x[batch_index].cpu().detach().numpy()
    #
    #             # 对每个通道创建heatmap
    #             plt.imshow(current_batch[0], cmap='hot', interpolation='nearest')
    #             plt.colorbar()
    #
    #             # 保存图片
    #             output_path = os.path.join(output_dir, f'{id[batch_index]}_channel_0.png')
    #             plt.savefig(output_path)
    #             plt.clf()  # 清除当前图像，准备绘制下一个
    #
    #             print(f' {id[batch_index]} heatmaps saved.')
    #
    #
    #         # 遍历每个batch
    #         # for batch_index in range(x.shape[0]):
    #         #     # 获取当前batch的特征图
    #         #     current_batch = x[batch_index].cpu().detach().numpy()
    #         #
    #         #     # 对每个通道创建heatmap
    #         #     for channel_index in range(current_batch.shape[0]):
    #         #         plt.imshow(current_batch[channel_index], cmap='hot', interpolation='nearest')
    #         #         plt.colorbar()
    #         #
    #         #         # 保存图片
    #         #         output_path = os.path.join(output_dir, f'{id[batch_index]}_channel_{channel_index}.png')
    #         #         plt.savefig(output_path)
    #         #         plt.clf()  # 清除当前图像，准备绘制下一个
    #         #
    #         #     print(f' {id[batch_index]} heatmaps saved.')
    #
    #         # print('All heatmaps saved.')
    #
    #
    #
    #
    #
    #     else:
    #
    #         x = self.conv1(x)
    #         x = self.bn1(x)
    #         x = self.relu(x)
    #
    #     x = self.maxpool(x)
    #
    #     c1 = self.layer1(x)
    #     c2 = self.layer2(c1)
    #     c3 = self.layer3(c2)
    #     c4 = self.layer4(c3)
    #
    #     return c1, c2, c3, c4
    # def base_forward(self, x, guide, val):
    #     def process_branch(x):
    #         x_fea_lst = [self.relu(self.bn1(self.conv1(x_i))) for x_i in x]
    #         x_stacked = torch.stack(x_fea_lst)
    #         x_sum = torch.sum(x_stacked, dim=0)
    #         x_processed = torch.sum(
    #             torch.stack([att_block(x_i, x_sum) for att_block, x_i in zip(self.att_blocks, x_fea_lst)]), dim=0)
    #         return x_processed
    #
    #     if val:
    #         x = process_branch(x)
    #     elif guide:
    #         x0, x1 = x
    #         x0 = process_branch(x0)
    #         x1 = process_branch(x1)
    #         x = torch.cat((x0, x1))
    #     else:
    #         x = self.relu(self.bn1(self.conv1(x)))
    #
    #     x = self.maxpool(x)
    #     c1 = self.layer1(x)
    #     c2 = self.layer2(c1)
    #     c3 = self.layer3(c2)
    #     c4 = self.layer4(c3)
    #
    #     return c1, c2, c3, c4
    def base_forward(self, x, guide, val, id):
        def process_input(x):
            x_sum = None
            x_fea_lst = []

            for xi in x:
                x_tmp = self.relu(self.bn1(self.conv1(xi)))
                x_fea_lst.append(x_tmp)
                if x_sum is None:
                    x_sum = x_tmp
                else:
                    x_sum = x_sum + x_tmp  # Avoid in-place operation

            x_att = None
            for att_block, x_i in zip(self.att_blocks, x_fea_lst):
                x_tmp = att_block(x_i, x_sum)
                if x_att is None:
                    x_att = x_tmp
                else:
                    x_att = x_att + x_tmp  # Avoid in-place operation

            return x_att

        if guide:
            x0, x1 = x
            x0 = process_input(x0)
            x1 = process_input(x1)
            x = torch.cat((x0, x1))
        elif val:
            x = process_input(x)

            # 创建保存图像的文件夹
            output_dir = 'heatmaps_Att'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 遍历每个batch
            for batch_index in range(x.shape[0]):
                # 获取当前batch的特征图
                current_batch = x[batch_index].cpu().detach().numpy()

                # 对每个通道创建heatmap
                plt.imshow(current_batch[0], cmap='hot', interpolation='nearest')
                plt.colorbar()

                # 保存图片
                output_path = os.path.join(output_dir, f'{id[batch_index]}_channel_0.png')
                plt.savefig(output_path)
                plt.clf()  # 清除当前图像，准备绘制下一个

                print(f' {id[batch_index]} heatmaps saved.')



        else:
            x = self.relu(self.bn1(self.conv1(x)))

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
    return model


def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)
