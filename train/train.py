import argparse

from config.config_v2 import conf
from model.initialization import initialization


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Train')
# todo 注意这里修改了cache的值，实际训练中最好打开
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the training data will be loaded at once'
                         ' before the training start. Default: TRUE')
opt = parser.parse_args()

m = initialization(conf, train=opt.cache)[0]

print("Training START")
m.fit()
print("Training COMPLETE")
