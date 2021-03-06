import ABS
import APoZ
import Geometric_median
import Loss_function
import network_slimming
import numpy as np
import pickle
import torch
import logging
import models1
import torch.optim as optim
import test
logger = logging.getLogger("__main__")
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#filter_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
#染色体长度是30，4224/50 = 84，则基因的每一位代表84个卷积核, 最后一条染色体代表84 + 24个filters
#所以通过某种方法得到的filters下标需要处理成50段

#处理filters组合，将其分为50段

def handle_code(pruning_filter, args):

    #print(pruning_filter)
    name = args.name
    gene_length = args.gene_length
    filter_num = args.filter_nums
    filter_length = sum(filter_num)
    pruning_filter_index = []
    
    index = 0
    count = 0

   # name 按顺序存储卷积层和全连接层的下标，按照name中的顺序取出层对应的filters
        
    for name_layer in name:
        temp_filters = pruning_filter[name_layer]
        if index == 0:
            for filters in temp_filters:
                pruning_filter_index.append(filters)
                
        else:
            count += filter_num[index - 1]
            for filters in temp_filters:
                temp = count + filters
                pruning_filter_index.append(temp)   
                
        index = index + 1
            
    pruning_filter_index = sorted(pruning_filter_index)
    #print(pruning_filter_index)
    

    
    #int是往下取整数，所以最后一个基因要存储多余出来的filters
    #或者往上取整可以尝试，向上取整基因长度可能小于30不能向上
    #pivot = int(filter_length / gene_length)
    #把基因的每一位初试化为1，1代表保留，0代表剪枝
    #这里表示每一列存储一个filters，为什么不是1行来存储？
    filter_code = np.ones((filter_length, 1), dtype=np.int)
    
    #出现在Pruning_filter_index中的filters的值被设为0
    for i in pruning_filter_index:
        filter_code[i] = 0
        
    #generate bn cfg
    temp = []
    cfg_code = []
    index = 0
    count = 0
    num = 0
    
    for i in filter_num:
        count += i
        if index == 0:
            temp = filter_code[: i]
            #print('0', len(temp))
            
        elif index == 39:
            temp = filter_code[count - i : count]
            #print('l', len(temp))

        elif index == 13 or index == 26:
            temp = filter_code[count - i : count]
            #print('l', len(temp))
            

        else:
            temp = np.concatenate((temp, filter_code[count - i : count]),axis=0)
            #temp.extend(filter_code[count - i : count])
            #print('l', len(temp))
            
            #print('t', temp)
        #print('l', len(temp))
        index += 1
        num += len(temp)

        cfg_code.append(temp)
    
    #num = 0
    #for i in cfg_code:
    #    print('len', len(i))
                               
        #print('sum', sum(np.array(i)))
        
    return cfg_code
 
def handle_cfg(filter_code, args):
    
    cfg_code = filter_code
    #for i in cfg_code:
    #    print('len', len(i))
                            
    gene_length = args.gene_length
    filter_num = args.bn_nums
    filter_length = sum(filter_num)
    pivot = int(filter_length / gene_length)

    cfg_code = [i for item in cfg_code for i in item]
    cfg_code = np.array(cfg_code)
    cfg_code = cfg_code.reshape(filter_length, 1)
    
    #print('cfg_code', cfg_code.shape)
    
    num = sum(np.array(cfg_code))
    #print('r', num/filter_length)
    #把filters化分成gene_length份，最后一位基因要存储所有剩下的filters
    #格式也是filters_nums * 1
    filter_divide = []
    
    j = 0
    for i in range(0, filter_length, pivot):
        if j == (gene_length - 1):
            low = j * pivot
            high = filter_length
            filter_divide.append(cfg_code[low : high])
            break

        else:
                low = j * pivot
                high = (j + 1) * pivot
                filter_divide.append(cfg_code[low : high])

        j = j + 1
        
    #generate bn cfg
    #print('filter_divide', np.array(filter_divide).shape)
    #print('filter_divide', np.array(filter_divide[16]).shape)

    return filter_divide

def load(args):
    print('load', load)
    if args.depth == 40:
        
        model = models1.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
        if args.cuda:
            checkpoint = torch.load('../logs/checkpoint_dense40.pth.tar')
            newdict = dict()
            for name, para in checkpoint['state_dict'].items():
                name = name.replace('module.','')
                newdict[name] = para  


        else:
            checkpoint = torch.load('../logs/checkpoint_dense40.pth.tar', map_location=torch.device('cpu'))   
            newdict = dict()
            for name, para in checkpoint['state_dict'].items():
                name = name.replace('module.','')
                newdict[name] = para  
                
    
    model.load_state_dict(newdict)
    
    return model

#得到100种filters组合，这里应该是初始化种群时用，对照时也需要用
def original_code(model, args, train_loader, test_loader, val_loader):
    
    model = load(args)
    #剪枝率范围：20% - 70%
    #剪枝方法：四种
    #选50个剪枝率，分别用四种方法剪枝，得到200个filters组合

    #表示组合的下标
    index = 1
    
    #存储函数接口
    #func_list = [APoZ.get_code, ABS.get_code, Geometric_median.get_code, Loss_function.get_code]
    
    pruning_filters = []
    #种群数量初始为200
    if args.cuda: 
        low = 0
        high = 50
        
    else:
        low = 30
        high = 31
        
    #for i in range(30, 31):
    for i in range(low, high):
        print(i)
        logger.info('low_high_{}'.format(i))
        #i = np.random.randint(20, 71)
        i = i / 100
        #print('ABS')
        
        
        pruning_filter = ABS.get_code(i, model, args, train_loader, test_loader, val_loader)
        
        #print('pruning_filter', pruning_filter)
        #temp = handle_code(pruning_filter, args)
        #print('pruning_filter', temp[0], temp[1])
        temp = np.array(handle_cfg(pruning_filter, args))
        pruning_filters.append(temp)
        
        
        #print('Geometric_median')
        pruning_filter = Geometric_median.get_code(i, model, args, train_loader, test_loader, val_loader)
        temp = handle_code(pruning_filter, args)
        
        temp = np.array(handle_cfg(temp, args))
        #print('temp', temp[0], temp[1])
        pruning_filters.append(temp)
        
        
        pruning_filter = network_slimming.get_code(i, model, args, train_loader, test_loader, val_loader)
        #print('pruning_filter', pruning_filter[0], pruning_filter[1])
        temp = np.array(handle_cfg(pruning_filter, args))
        pruning_filters.append(temp)
        
        #print('APoZ')
        pruning_filter = APoZ.get_code(i, model, args, train_loader, test_loader, val_loader)
        temp = handle_cfg(pruning_filter, args)
        #print('pruning_filter', pruning_filter[0], pruning_filter[1])
        #print(temp)
        pruning_filters.append(temp)
                

        
    pickle.dump(pruning_filters, open('filter_dict/filter_dict_densenet%s.pkl' % (args.depth), 'wb'))  








