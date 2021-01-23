#模型采用的是VGG16norm，每一层后面都有一个batchnorm操作，对于全连接层只有两层；
#可能是因为查看的这两个算法都只对卷积层剪枝，所以全连接层不重要可以改变
#先这样做，等会改回来
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy



#计算absolute weight sum（ABS），排序，取每层值最小的三个filters

def DRIVE(filter_list, num_t):
    
    #abs_filter = [np.sum(np.abs(i)) for item in filter_list for i in item]
    abs_filter = [np.sum(np.abs(p)) for p in filter_list]
    small_rank = np.argsort(abs_filter)

    return small_rank[: num_t]

#conv1 1*10*5*5 输入通道是1，输出通道是10
def absolute_weight_sum_conv(conv_name, para_data, res, num_t, index):
    
    conv_name = conv_name.split('.', 4)
    name = str(index)
    #if 'weight' in conv_name:
    p = copy.deepcopy(para_data)
    #p = p.squeeze()
    filter_list = [i.squeeze().data.cpu().numpy() for item in para_data for i in item]
    #print('len', len(filter_list))
    
    #问题是对于位数只有1和超过2的卷积层统计会出错
    res[name] = DRIVE(filter_list, num_t)

def absolute_weight_sum_fc(fc_name, para_data, res, num_t, index):
    
    fc_name = fc_name.split('.', 4)
    name = str(index)
    #if 'weight' in conv_name:
    p = copy.deepcopy(para_data)
    p = p.squeeze()
    filter_list = [p[i].squeeze().data.cpu().numpy() for i in range(p.size(0))]

    res[name] = DRIVE(filter_list, num_t)

def handle_cfg(pruning_filter, args):

    #print(pruning_filter)
    #name = args.name
    filter_num = args.bn_nums
    filter_length =sum(filter_num)
    cfg = []
    res = []
    for i in filter_num:
        temp = [1] * i
        cfg.append(temp)
    
    for i in range(len(cfg)):
        temp = cfg[i]
        name = str(i)
        index = pruning_filter[name]
        for ind in index:
            temp[ind] = 0
        res.append(temp)
        
    #print(res)    
        
    return res

def get_code(compress_rate, model, args, train_loader, test_loader, val_loader):
    res_ABS = {}
    #model.named_parameters.cpu()
    index = 0
    temp = []

    for name, para in model.named_parameters():
        
        #print(name, para.shape)
        if para.requires_grad:
            if 'weight' in name:
                #print('para', para.shape)
                #如果是卷积层且参数维度为4
                if para.dim() == 4:
                     
                    if index == 13 or index == 26:
                        temp[:] = []
                        temp.append(para)
                        
                    else:
                        temp.append(para)
                      
                        
                    #print('cat', temp[index].shape)
                
                    num_t = int(args.bn_nums[index] * compress_rate)
                    absolute_weight_sum_conv(name, temp, res_ABS, num_t, index)
                    index = index + 1

     
    #print(res_ABS)
    res_ABS[str(index)] = [1] * args.fc_nums
    #print(index)
        
            
    return handle_cfg(res_ABS, args)


