# -*- coding: utf-8 -*-
# 作者      ：wanglin   
# 创建时间   ：2020/9/28  9:38 
# 文件      ：demo2
# IDE      ：PyCharm
import numpy as np
import visdom

if __name__ == '__main__':
    """vis = Visdom(env="GaitSet_test", log_to_filename="work/log/visdom/test_all_acc.log")
    acc_array_exclude = np.loadtxt("work/log/visdom/acc_array_exclude.txt")
    acc_array_include = np.loadtxt("work/log/visdom/acc_array_include.txt")
    iter_list = np.loadtxt("work/log/visdom/iter_list.txt")
    print(acc_array_exclude.shape)
    print(iter_list.shape)
    iter_list_new = list(filter(lambda x: 0 == x % 100, iter_list.tolist()))
    print(iter_list_new.__len__())
    vis.bar(X=acc_array_exclude,
            opts=dict(
                stacked=False,
                legend=['NM', 'BG', 'CL'],
                rownames=iter_list_new,
                title='Test_acc exclude',
                ylabel='rank-1 accuracy',  # y轴名称
                xtickmin=0.4  # x轴左端点起始位置
                # xtickstep=0.4  # 每个柱形间隔距离
            ), win="acc_array_exclude")
    vis.bar(X=acc_array_include,
            opts=dict(
                stacked=False,
                legend=['NM', 'BG', 'CL'],
                rownames=iter_list_new,
                title='Test_acc include',
                ylabel='rank-1 accuracy',  # y轴名称
                xtickmin=0.4,  # x轴左端点起始位置
                # xtickstep=0.4  # 每个柱形间隔距离,
                append=True
            ), win="acc_array_include")"""

    track_loss = 0  # for draw graph
    global_step = 0
    vis = visdom.Visdom(env=u"train_loss")
    win = vis.line(X=np.array([global_step]), Y=np.array([track_loss]))

    for epoch in range(10):
        # 此处省略代码

        for iter_num, dial_batch in enumerate(range(20)):
            # 此处省略代码
            loss = np.random.random()
            vis.line(X=np.array([global_step]), Y=np.array([loss]), win=win,
                     update='append', opts=dict(title="demo"))  # for draw graph
            global_step += 1
