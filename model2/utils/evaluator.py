import numpy as np
import torch
import torch.nn.functional as F


def cuda_dist(x, y):
    # 计算x中的每个样本和y中每个样本的距离
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    # 返回的形状为：x.size(0) * y.size(0)
    return dist


def evaluation(data, config):
    # data : np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    # 下面的循环是求出probe在probe_view视角下，gallery视角在gallery_view的准确率，而且在是在probe_seq下和对应的gallery_seq下的,
    # probe_seq因为包含三种行走条件下的
    #                   集合个数                      视角个数  视角个数   top5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):  # probe集合
        for gallery_seq in gallery_seq_dict[dataset]:  # gallery集合
            for (v1, probe_view) in enumerate(view_list):  # probe视角列表
                for (v2, gallery_view) in enumerate(view_list):  # gallery视角列表
                    # seq(NM-01,NM-02...)类型元素在gallery_seq中，并且在当前的gallery_view 中，因为要求每个视角下的准确率
                    # gallery_seq和probe_seq都是列表
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]  # 找出对应的gallery样本的特征
                    gallery_y = label[gseq_mask]  # 找出对应的gallery样本的标签
                    # 下面的类似。找出相应的probe的样本特征，标签等
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    idx = dist.sort(1)[1].cpu().numpy()  # 对probe中的每个样本的预测的结果进行排序，这里返回的是在原始数组中的下标，
                    acc[p, v1, v2, :] = np.round(  # 这里相当于在计算top(num_rank)的准确率
                        # acc[p, v1, v2, 0]保存的是top1准确率，而acc[p, v1, v2, num_rank-1]保存的是top5准确率（因为这里的num_rank=5)
                        # gallery_y[idx[:, 0:num_rank] 按下标取出前num_rank个样本标签
                        # 注意这里计算的是top(num_rank)的准确率，
                        # np.cumsum做一个累计计算，计算top_1,top_2,...,top_num_rank的准确率
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc
