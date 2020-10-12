import os
import os.path as osp

import cv2
import numpy as np
import torch.utils.data as tordata
import xarray as xr


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution) / 64 * 10)  # 10
        self.data_size = len(self.label)  # 数据集样本个数
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)  # 去重 ，保存所有的人物标签
        self.seq_type_set = set(self.seq_type)  # 去重，保存最终的种类（bg-01。。。）
        self.view_set = set(self.view)  # 视角种类
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1  # 如果有些轮廓序列缺失，那么其在index_dict中用-1表示其不存在
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])
        # 用来存储每个样本的对应的下标信息，将其对应到这个三维数组中去

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i
            # 将所有的样本的下标信息(在self.label，self.seq_type，self.view中的下标信息进行保存)进行保存

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        """
        一个样本的大小为
        `30 * 64 * 64`，然后进行了一个裁剪，对宽度进行了裁剪，处理后的大小为
        `32 * 64 * 44`
        """
        return self.img2xarray(path)[:, :, self.cut_padding:-self.cut_padding]. \
                   astype('float32') / 255.0

    def __getitem__(self, index: int):
        # pose sequence sampling
        # 不使用cache的情况下，直接返回index下标的数据，否则将如果index数据之前没有读取过，就将其加载到self.data中进行缓存，下次用到直接读取，不用重新从磁盘中进行读取
        if not self.cache:
            # 加载index样本的所有的轮廓剪影图片，例如，_path:/data/lwl/Gait_experiment/gait_data/002/bg-01/000
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]  # 取出对应的帧序号组成集合
            frame_set = list(set.intersection(*frame_set))  # 返回集合交集
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]
        # TODO 打印data的大小,以及真正的帧的大小
        # print("data的大小为：", len(data))
        # print(data[0].shape)
        return data, frame_set, self.view[index], self.seq_type[index], self.label[index],

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        # 读取指定路径下的所有轮廓剪影，并且将其缩放到64*63*1大小，[:, :, 0]最后切片取出为一个矩阵64*64
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)), [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]

        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],  # 帧编号，帧高，帧宽
        )
        return data_dict

    def __len__(self):
        return len(self.label)
