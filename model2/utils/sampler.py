import random

import torch.utils.data as tordata


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()
            pid_list = random.sample(  # 选出相应的p（batch_size[0])个人，这里设置的是选取8个人
                list(self.dataset.label_set),
                self.batch_size[0])
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index > 0].flatten().tolist()  # 将那些存在轮廓信息的样本的下标取出来，因为下标为-1说明其轮廓序列不存在
                _index = random.choices(
                    _index,  # 从每个人的样本集合中选出k（batch_szie[1])个轮廓序列，这里设置的是16个
                    k=self.batch_size[1])
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
