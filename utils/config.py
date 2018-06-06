#coding:utf8
import warnings
class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'UNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test1'  # 测试集存放路径
    load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载
    model_path = "models/V1.0/U_Net.pkl"  # 模型的地址
    path_folder = ""

    n_channel = 3  # 输入图片维数
    n_class = 2  # 图片分类

    batch_size = 1  # batch size
    batch_num = 36038
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    epoch_num = 10
    lr = 2e-5  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse