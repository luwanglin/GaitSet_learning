import numpy as np
import torch
import torch.nn as nn

from .basic_blocks_dyrelu import SetBlock, BasicConv2d


class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        # set_layer1.set_layer2 对应于c1,c2 P
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        # set_layer3.set_layer4 对应于c3,c4 P
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))
        # set_layer4.set_layer5 对应于c5,c6 P

        _gl_in_channels = 32
        _gl_channels = [64, 128]
        # 和上面的结构相同，两个3*3卷积加上池化层
        self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)
        # 这里也是一样
        self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)

        self.bin_num = [1, 2, 4, 8, 16]  # 论文中的五个尺度在HPM中的
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(  # 参数的形状为62*128*256
                    torch.zeros(sum(self.bin_num) * 2, 128, hidden_dim)))])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    def forward(self, silho, batch_frame=None):

        # TODO 注意这里为了进行可视化网络结构，修改了部分代码
        # batch_frame = silho[1]
        # silho = silho[0]
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel

        if batch_frame is not None:
            # 取出了第一个GPU的batch数据
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                # 找出第一个帧数不为0的样本，因为之前batch_frame进行了填充0
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]  # 排除掉填充的样本
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        # silho:128*30*64*44
        n = silho.size(0)
        # x:128*30*1*64*44
        x = silho.unsqueeze(2)

        x = self.set_layer1(x)
        x = self.set_layer2(x)
        # self.frame_max()的返回值为 torch.Size([128, 32, 32, 22])
        # 这里的self.frame_max相当于set pooling 采用了max统计函数
        gl = self.gl_layer1(self.frame_max(x)[0])
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)

        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x)[0])
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)[0]
        gl = gl + x

        feature = list()
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:  # 这里的循环相当于对feature map运用HPP
            z = x.view(n, c, num_bin, -1)  # 按高度进行划分成strips
            z = z.mean(3) + z.max(3)[0]  # 应用maxpool和avgpool
            feature.append(z)  # z的形状为 n,c,num_bin
            z = gl.view(n, c, num_bin, -1)  # 对gl也运用HPP
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)  # 将gl和z的都加入到feature中
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()

        # 由于不同比例尺度上的条带描绘了不同的感受野特征，并且每个比例尺度上的不同条带描绘了不同的空间位置的特征，因此使用独立的FC很自然的
        # feature：62*128*128，self.fc_bin:62*128*256
        # 相当于62个条带，每个条带128维，那么对每个条带分别进行FC的映射
        feature = feature.matmul(self.fc_bin[0])
        # 这样经过全连接层计算之后就变成了 62*128*256
        feature = feature.permute(1, 0, 2).contiguous()
        # 维度变换，128*62*256

        return feature, None
