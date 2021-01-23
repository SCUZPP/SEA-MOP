class Args:
    def __init__(self):
    
        self.depth = 50
        #kgea = False
        self.constraint = False
        
        #此rand是指对两种不对全连接层剪枝的算法随机产生剪枝下标
        self.rand = False
        self.dataset = 'imagenet'
        self.arch = 'resnet50'
        #指重新训练
        self.retrain = False
        self.cpu = False
        self.half = False
        self.print_freq = 1
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.1
        self.epochs = 300
        self.batch_size = 256
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
       
        self.ft_epochs = 100
        self.re_epochs = 100
        self.temp_epoch = 60
        self.filter_nums = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048, 1000]
        self.name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49']
        self.gene_length = 50
        self.filter_length = sum(self.filter_nums)
        self.accuracy = 94.48

        