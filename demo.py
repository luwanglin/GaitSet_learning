# -*- coding: utf-8 -*-
# 作者      ：wanglin   
# 创建时间   ：2020/8/29  16:40  
# 文件      ：demo
# IDE      ：PyCharm   创建文件的IDE名称
import numpy
from visdom import Visdom

if __name__ == '__main__':
    '''vis = Visdom(env="demo", log_to_filename="./visdom.log")
    print(len(np.random.rand(20, 3)), len(np.arange(0, 20)))
    vis.bar(
        X=np.random.rand(20, 3),
        Y=np.arange(0, 20),
        opts=dict(
            stacked=False,
            legend=['The Netherlands', 'France', 'United States']
        )
    )'''
    # Visdom.replay_log(log_filename="./visdom.log")

    vis = Visdom(env="test")
    acc_array_exclude = numpy.array([[87.61818181818184, 78.15718181818183, 64.21818181818182],
                                     [87.55454545454546, 78.20354545454546, 64.36363636363637]])
    iter_list = [88800, 88900]
    # numpy.savetxt("work/log/visdom/acc.txt", acc_array_exclude)
    # numpy.savetxt("work/log/visdom/iter_list.txt", iter_list)
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
    vis.bar(X=acc_array_exclude,
            opts=dict(
                stacked=False,
                legend=['NM', 'BG', 'CL'],
                rownames=[88901, 88902],
                title='Test_acc',
                ylabel='rank-1 accuracy',  # y轴名称
                xtickmin=0.4,  # x轴左端点起始位置
                # xtickstep=0.4  # 每个柱形间隔距离
            ), win="acc_array_exclude",)
