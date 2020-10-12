conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "/data/lwl/Gait_experiment/gait_data",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,  # LT划分方式 74用于训练（In CASIA-B, data of subject #5 is incomplete.），其余的用于测试
        'pid_shuffle': False,
        # 是否进行随机的划分数据集，如果为False，那么直接选取1-74为训练集，剩余的测试集
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        # TODO 注意这里修改了batchsize的的大小
        'batch_size': (8, 4),
        # TODO 注意这里修改了接着训练的轮数
        'restore_iter': 7690,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 8,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
