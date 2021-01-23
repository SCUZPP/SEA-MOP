import numpy as np
import copy
import torch
import torch.nn as nn
from models import *
import copy
#self.filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
def masking(solution, args):
    
    filter_nums = args.bn_nums
    solution = np.squeeze(solution)

        
        
    res = []
    low = 0
    count = []
    c = 0
    high = filter_nums[0]
    #print('solution', solution.shape)
    for index in range(len(filter_nums)):
        
        if index == 0:
            filters = solution[low : high]
            
            
        elif index == len(filter_nums) - 1:
             
            c = c + filter_nums[len(filter_nums) - 1]      
            break
            
        else:
            low = high
            high = filter_nums[index] + low
            filters = solution[low : high]
            
        #if index == 0 or index == 1:   
        #    print('len_filters', filters)
            
        temp = np.sum(filters)
        #print('temp', temp)
        res.append(filters)
        count.append(temp)        
        c = c + temp
            
 
    #fc 
    print('count', count)
    return res, count, c
    
def prune_model(model_original, solution, args):
    model = copy.deepcopy(model_original)
    #newmodel = copy.deepcopy(model_original)
    
    #把一维矩阵根据卷积层改为二维矩阵
    cfg_mask, cfg, solution_new = masking(solution, args)
    #print('cfg', cfg)
    #print('l', len(cfg))
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    conv_count = 1
        
    newmodel = densenet(dataset=args.dataset, depth=args.depth, cfg=cfg)
    #print('oldmodel', model)
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    #print('middle',num_parameters)

    if args.cuda:
        newmodel.cuda()
 
    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    first_conv = True

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batch normalization layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the mask parameter `indexes` for the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.copy()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
                continue

        elif isinstance(m0, nn.Conv2d):
            if first_conv:
                # We don't change the first convolution layer.
                m1.weight.data = m0.weight.data.clone()
                first_conv = False
                continue
            if isinstance(old_modules[layer_id - 1], channel_selection):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
                #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                # If the last layer is channel selection layer, then we don't change the number of output channels of the 
                # current convolutional layer.
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    num_parameters1 = sum([param.nelement() for param in newmodel.parameters()])
    num_parameters2 = sum([param.nelement() for param in model.parameters()])
    #print('1', num_parameters1)
    #print('2', num_parameters2)
    #print(newmodel)
    #print('m', model)
    
        
    return newmodel, solution_new