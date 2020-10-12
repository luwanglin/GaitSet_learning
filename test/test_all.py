import argparse
import os
import sys
from datetime import datetime

sys.path.append(r"/data/lwl/Gait_experiment/GaitSet")
# sys.path.append(r"/data/lwl/Gait_experiment/GaitSet/model")
import numpy as np
# from config.config_v2 import conf
from config.config_v2 import conf
from visdom import Visdom
from model2.initialization import initialization
from model2.utils import evaluation


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0  # 本来11个视角，除去相同视角还有10个
    if not each_angle:
        result = np.mean(result)
    return result


if __name__ == '__main__':
    m = initialization(conf, test=opt.cache)[0]
    """print(os.listdir(os.path.join(conf['WORK_PATH'], "checkpoint",
                                  conf["model"]["model_name"]))[-1].split("-")[-2])
    test1 = []
    for filename in os.listdir(os.path.join(conf['WORK_PATH'], "checkpoint",
                                            conf["model"]["model_name"])):
        if int(filename.split("-")[-2]) == 60100:
            test1.append(filename)
    print(test1)"""

    # 找出所有保存的模型的对应的代数,记得去重(因为和之前的128batchsize大小的重复了，那个没有采用dy-relu)
    iter_list = sorted(list(set(map(lambda a: int(a.split("-")[-2]),
                                    os.listdir(os.path.join("checkpoint",
                                                            conf["model"]["model_name"]))))))
    # writer = SummaryWriter(log_dir="./log/all_test_acc_log")

    # visdom可视化
    vis = Visdom(env="GaitSet_test", log_to_filename="./log/visdom/test_all_acc.log")

    # load model checkpoint of iteration opt.iter
    acc_array_include = []
    acc_array_exclude = []
    # 只取出大于60000的进行测试
    iter_list = list(filter(lambda x: x > 70000, iter_list))
    for iter_s in iter_list:
        if iter_s % 100 == 0:
            print(iter_s)
            print('Loading the model of iteration %d...' % iter_s)
            m.load(iter_s)
            print('Transforming...')
            time = datetime.now()
            test = m.transform('test', opt.batch_size)
            print('Evaluating...')
            acc = evaluation(test, conf['data'])
            print('Evaluation complete. Cost:', datetime.now() - time)

            # Print rank-1 accuracy of the best model
            # e.g.
            # ===Rank-1 (Include identical-view cases)===
            # NM: 95.405,     BG: 88.284,     CL: 72.041

            # 我训练得到的
            # ===Rank-1 (Include identical-view cases)===
            # NM: 95.744,	BG: 89.143,	CL: 72.554

            for i in range(1):
                print('===Rank-%d (Include identical-view cases)===' % (i + 1))
                print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                    np.mean(acc[0, :, :, i]),
                    np.mean(acc[1, :, :, i]),
                    np.mean(acc[2, :, :, i])))
                acc_array_include.append([np.mean(acc[0, :, :, i]),
                                          np.mean(acc[1, :, :, i]),
                                          np.mean(acc[2, :, :, i])])

            # Print rank-1 accuracy of the best model，excluding identical-view cases
            # e.g.
            # ===Rank-1 (Exclude identical-view cases)===
            # NM: 94.964,     BG: 87.239,     CL: 70.355

            # 我训练得到的
            # ===Rank-1 (Exclude identical-view cases)===
            # NM: 95.327,	BG: 88.221,	CL: 70.745
            for i in range(1):
                print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
                print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                    de_diag(acc[0, :, :, i]),
                    de_diag(acc[1, :, :, i]),
                    de_diag(acc[2, :, :, i])))
                acc_array_exclude.append([de_diag(acc[0, :, :, i]),
                                          de_diag(acc[1, :, :, i]),
                                          de_diag(acc[2, :, :, i])])
    print("这是acc数组：", acc_array_exclude)
    print("这是rownames：", iter_list)
    # 保存数据
    np.savetxt("./log/visdom/acc_array_exclude.txt", np.array(acc_array_exclude))
    np.savetxt("./log/visdom/acc_array_include.txt", np.array(acc_array_include))
    np.savetxt("./log/visdom/iter_list.txt", iter_list)
    vis.bar(X=acc_array_exclude,
            opts=dict(
                stacked=False,
                legend=['NM', 'BG', 'CL'],
                rownames=iter_list,
                title='Test_acc',
                ylabel='rank-1 accuracy',  # y轴名称
                xtickmin=0.4  # x轴左端点起始位置
                # xtickstep=0.4  # 每个柱形间隔距离
            ), win="acc_array_exclude")
    vis.bar(X=acc_array_include,
            opts=dict(
                stacked=False,
                legend=['NM', 'BG', 'CL'],
                rownames=iter_list,
                title='Test_acc',
                ylabel='rank-1 accuracy',  # y轴名称
                xtickmin=0.4  # x轴左端点起始位置
                # xtickstep=0.4  # 每个柱形间隔距离
            ), win="acc_array_include")
