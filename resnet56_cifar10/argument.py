class Args:
    def __init__(self):
    
        self.depth = 56
        #kgea = False
        self.constraint = False
        
        #此rand是指对两种不对全连接层剪枝的算法随机产生剪枝下标
        self.rand = False
        self.dataset = 'cifar10'
        self.arch = 'resnet'
        #指重新训练
        self.retrain = False
        self.cpu = True
        self.half = False
        self.print_freq = 20
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.1
        self.epochs = 300
        self.batch_size = 64
        self.test_batch_size = 500
        self.val_batch_size = 500
        self.momentum = 0.9
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.weight_decay = 1e-4
        #最后三层是全连接层
        if self.depth == 56:
        
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 100]
            self.fc_nums_dict = { '55': 100}
            self.conv_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
            self.fc_layer = [55]
            self.conv_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']
            self.name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55']
            self.fc_name = ['55']
            self.conv_name_dict = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': [], '12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': [], '19': [], '20': [], '21': [], '22': [], '23': [], '24': [], '25': [], '26': [], '27': [], '28': [], '29': [], '30': [], '31': [], '32': [], '33': [], '34': [], '35': [], '36': [], '37': [], '38': [], '39': [], '40': [], '41': [], '42': [], '43': [], '44': [], '45': [], '46': [], '47': [], '48': [], '49': [], '50': [], '51': [], '52': [], '53': [], '54': []}
            self.fc_name_dict = { '55': []}
            self.gene_length = 50
            self.filter_length = 2132
            self.accuracy = 94.080
        
        