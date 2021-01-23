class Args:
    def __init__(self):
    
        self.depth = 18
        #kgea = False
        self.constraint = False
        
        #此rand是指对两种不对全连接层剪枝的算法随机产生剪枝下标
        self.rand = False
        self.dataset = 'imagenet'
        self.arch = 'resnet18'
        #指重新训练
        self.retrain = False
        self.cpu = False
        self.half = False
        self.print_freq = 1
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.1
        self.epochs = 300
        self.batch_size = 64
        self.test_batch_size = 1024
        self.val_batch_size = 1024
        self.momentum = 0.9
        self.no_cuda = False
        self.cuda = True
        self.workers = 16
        self.seed = 1
        self.log_interval = 10
        self.weight_decay = 1e-4
        #最后三层是全连接层
       
        self.ft_epochs = 300
        self.re_epochs = 300
        self.temp_epoch = 60
        self.filter_nums = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 1000]
        self.name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
        self.gene_length = 50
        self.filter_length = sum(self.filter_nums)
        self.accuracy = 94.48

        