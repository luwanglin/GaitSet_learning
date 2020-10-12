import os
import os.path as osp

import numpy as np

from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()  # 存放的一个样本的路径地址（因为GaitSet中一个样本是一个轮廓剪影的集合），存放轮廓序列的地址，
    # 如：/data/lwl/Gait_experiment/gait_data/001/bg-01/000
    view = list()  # 存放样本的视角标签，即000，018,...,180，注意这里存放的是和上面样本对应的视角信息
    seq_type = list()  # 存放样本的序列标记信息，即bg-01,和上面一样对应于每个样本
    label = list()  # 存放的是样本的ID信息，与每个样本分别对应

    for _label in sorted(list(os.listdir(dataset_path))):  # 遍历人物ID标签
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):  # 遍历人物的轮廓序列类型
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):  # 遍历轮廓序列的视角
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)  # 遍历出所有的轮廓剪影
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)

    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))
    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)  # 是否对数据集进行随机的划分，注意的是第5个元素被忽略了
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)
        # 存放训练集测试集的划分，包括训练集和测试集的人物ID号，第一部分是训练集，第二部分是测试集

    pid_list = np.load(pid_fname, allow_pickle=True)
    train_list = pid_list[0]
    test_list = pid_list[1]
    train_source = DataSet(
        # 存放训练集样本的路径地址
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        # 存放的是训练集样本的标签
        [label[i] for i, l in enumerate(label) if l in train_list],
        # 训练集样本的序列类型 如：bg-01之类
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        # 训练集样本对应的视角信息
        [view[i] for i, l in enumerate(label) if l in train_list],
        cache,
        resolution)
    # 以下同上存放的是测试集的相关样本信息
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution)

    return train_source, test_source
