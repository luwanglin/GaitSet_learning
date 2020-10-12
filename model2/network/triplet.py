import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, feature, label):
        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.size()
        # hp_mask是找出所有样本对中具有相同标签的，相同的为true，不同的为false
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        # hn_mask与上面相反，是找出不同的标签的样本对
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)
        # 62*128*128
        dist = self.batch_dist(feature)  # 这里求出了batch中每个样本的各个条带之间的欧式距离
        # mean_dist:62
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)
        # 这里是困难样本对发掘，找出每个样本对应的正样本对中的最大距离，找出每个样本的每个负样本对中最小距离，这就相对于进行困难样本挖掘
        # hard
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)
        # 计算每个条带的hard_loss的平均值
        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)

        # 这里是求取所有正负样本对的loss，没有进行困难样本挖掘
        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
        # 计算每个正样本对和负样本对之间的triplet loss
        # full_loss_metric_sum:62
        full_loss_metric_sum = full_loss_metric.sum(1)
        # 对每个条带中loss不为0的样本进行统计
        full_loss_num = (full_loss_metric != 0).sum(1).float()  # loss不为0的进行计数
        # 计算每个条带的所有triple loss平均值
        full_loss_metric_mean = full_loss_metric_sum / full_loss_num  # loss不为0的样本才贡献了损失，所以只对贡献的样本进行平均
        full_loss_metric_mean[full_loss_num == 0] = 0
        # 返回值的形状依次为：62 ,            62,                  62,        62
        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):
        # x:[62, 128, 256]
        # 相当于：d(A,B)=A^2+B^2-2*A*B,这里采用批量的方式求取了每个样本之间的距离
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        # 62*128*128
        return dist
