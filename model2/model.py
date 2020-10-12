import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
from tensorboardX import SummaryWriter
from visdom import Visdom

from .network import TripletLoss, SetNet
from .utils import TripletSampler


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name: str,
                 train_source,
                 test_source,
                 img_size=64,
                 logdir="./log",
                 model_save_dir="GaitSet"):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source
        self.model_save_dir = model_save_dir

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        self.encoder = SetNet(self.hidden_dim).float()
        # TODO 这里修改了多卡运行的代码
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'
        self.logdir = logdir

    def collate_fn(self, batch):

        batch_size = len(batch)  # batch的大小
        """
        data = [self.__loader__(_path) for _path in self.seq_dir[index]]
        feature_num代表的是data数据所包含的集合的个数,这里一直为1，因为读取的是
          _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)  # 遍历出所有的轮廓剪影
        """
        feature_num = len(batch[0][0])

        seqs = [batch[i][0] for i in range(batch_size)]  # 对应于data
        frame_sets = [batch[i][1] for i in range(batch_size)]  # 对应于 frame_set
        view = [batch[i][2] for i in range(batch_size)]  # 对应于view[index]
        seq_type = [batch[i][3] for i in range(batch_size)]  # 对应于self.seq_type[index]
        label = [batch[i][4] for i in range(batch_size)]  # 对应于self.label[index]
        batch = [seqs, view, seq_type, label, None]

        '''
         这里的一个样本由 data, frame_set, self.
         view[index], self.seq_type[index], self.label[index]
         组成
        '''

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                # 这里的random.choices是有放回的抽取样本
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
                # feature.loc[]传入list会取出一组的数据
            else:
                # 或者选取所有的帧
                _ = [feature.values for feature in sample]
            return _

        # 提取出每个样本的帧，组成list，存的是array组成的list
        seqs = list(map(select_frame, range(len(seqs))))
        # print(self.sample_type, "采样版本")
        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]

        else:
            # TODO 这里更改了GPU的个数
            gpu_num = min(torch.cuda.device_count(), batch_size)
            # gpu_num = min(4, batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)

            # batch_frames的内容：
            # [[gpu1_sample_1_frameNumbers,gpu1_sample_2_frameNumbers,...],[gpu2_sample_1_frameNumbers,gpu2_sample_2_frameNumbers,...],....]
            batch_frames = [[  # 将数据划分到不同的GPU上
                len(frame_sets[i])  # 每个样本的帧的总数数
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)  # 最后一个GPU上的batch大小不够时，补0
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])  # 求出哪个GPU上的帧最多

            # 将每个GPU上的对应的batch数据进行拼接，组成最终的一个大的array
            # seqs=[[gpu1_batch,gpu2_batch,gpu3_batch,.....]]
            seqs = [[
                np.concatenate([  # 这里将一个batch所有的帧进行了拼接
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            # TODO 打印seqs[j][_]的形状大小
            # print("seqs的大小：", seqs[0][0].shape)
            # 此时的
            seqs = [np.asarray([
                np.pad(seqs[j][_],  # seqs[j][_]的大小为(GPU_batch_size*frame_number)*64*44
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',  # 将每个batch的总帧数像最多帧数看齐，如果不够则补0
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]

            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        # TODO 打印seqs的形状大小
        # print("seqs的形状大小：", seqs[0].shape)
        return batch

    def fit(self):
        torch.backends.cudnn.benchmark = True
        writer = SummaryWriter(self.logdir)
        vis = Visdom(env="hard_full_loss", log_to_filename=osp.join(self.logdir, "hard_full_loss.log"))
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        # todo 这里改变了采样的方式
        # self.sample_type = 'all'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)  # 自定义的采样函数
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()
            # TODO 这里修改了将数据放在CPU上
            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # with SummaryWriter(comment="encoder") as w:
            #     self.encoder.cpu()
            #     # seq.cpu()
            #     # batch_frame.cpu()
            #     w.add_graph(self.encoder, (seq[0],))

            # todo 这里在退出程序，可视化encoder网络结构
            # sys.exit()
            # print("seq:", seq)
            # feature：128*62*256
            feature, label_prob = self.encoder(*seq, batch_frame)
            # 存放的是在train_label_set = list(self.train_source.label_set)中的下标位置信息
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            # 这里维度变换之后变成了 62*128*256
            triplet_feature = feature.permute(1, 0, 2).contiguous()
            # triplet_label:62*128
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            loss = 0
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()
            # todo 增加了loss的值
            loss += hard_loss_metric.mean()
            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:  # 打印每隔1000代的训练时间
                print(datetime.now() - _time1)
                _time1 = datetime.now()
            # TODO 更改了每10个batch打印一次
            if self.restore_iter % 10 == 0:
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                writer.add_scalar("hard_loss_metric", np.mean(self.hard_loss_metric), self.restore_iter)
                vis.line(X=np.array([self.restore_iter]), Y=np.array([np.mean(self.hard_loss_metric)]),
                         win="hard_loss_metric",
                         update="append",
                         opts=dict(title="hard_loss_metric"))

                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                writer.add_scalar("full_loss_metric", np.mean(self.full_loss_metric), self.restore_iter)
                vis.line(X=np.array([self.restore_iter]), Y=np.array([np.mean(self.full_loss_metric)]),
                         win="full_loss_metric",
                         update="append",
                         opts=dict(title="full_loss_metric"))

                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                writer.add_scalar("full_loss_num", np.mean(self.full_loss_num), self.restore_iter)
                vis.line(X=np.array([self.restore_iter]), Y=np.array([np.mean(self.full_loss_num)]),
                         win="full_loss_num",
                         update="append",
                         opts=dict(title="full_loss_num"))

                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                writer.add_scalar("mean_dist", self.mean_dist, self.restore_iter)
                vis.line(X=np.array([self.restore_iter]), Y=np.array([self.mean_dist]), win="mean_dist",
                         update="append",
                         opts=dict(title="mean_dist"))

                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], self.restore_iter)
                vis.line(X=np.array([self.restore_iter]), Y=np.array([self.optimizer.param_groups[0]['lr']]), win="lr",
                         update="append",
                         opts=dict(title="lr"))
                vis.line(X=np.array([self.restore_iter]), Y=np.array([loss.data.cpu().numpy()]), win="all_loss",
                         update="append",
                         opts=dict(title="all_loss"))
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []
            if self.restore_iter % 100 == 0:
                self.save()

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        # TODO 这里修改了不让数据到GPU上
        return autograd.Variable(x).cuda()
        # return autograd.Variable(x)

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        with torch.no_grad():
            self.encoder.eval()
            source = self.test_source if flag == 'test' else self.train_source
            self.sample_type = 'all'
            data_loader = tordata.DataLoader(
                dataset=source,
                batch_size=batch_size,
                sampler=tordata.sampler.SequentialSampler(source),
                collate_fn=self.collate_fn,
                num_workers=self.num_workers)

            feature_list = list()
            view_list = list()
            seq_type_list = list()
            label_list = list()

            for i, x in enumerate(data_loader):
                seq, view, seq_type, label, batch_frame = x
                for j in range(len(seq)):
                    seq[j] = self.np2var(seq[j]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()
                # print(batch_frame, np.sum(batch_frame))

                feature, _ = self.encoder(*seq, batch_frame)
                n, num_bin, _ = feature.size()
                feature_list.append(feature.view(n, -1).data.cpu().numpy())
                view_list += view
                seq_type_list += seq_type
                label_list += label
                # feature_list中每个元素的形状为：128*（62X256)=128*15872
                # 返回所有样本的特征向量组成的数组，形状为：样本总数*15872
            return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_save_dir), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_save_dir,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_save_dir,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.model_save_dir, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_save_dir,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
