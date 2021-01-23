import numpy as np
import copy
import torch
import torch.nn as nn
from models import *

#self.filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
def masking(solution, args):
    
    filter_nums = args.filter_nums
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
        else:
            low = high
            high = filter_nums[index] + low
            filters = solution[low : high]#[0]
        temp = np.sum(filters)
        
        if index == 0:
            c = c + temp

        if index%2 != 0 and index < len(filter_nums) - 1:
            #print(index)
            res.append(filters)
            count.append(temp)
            c = c + temp
            c = c + temp
            
        if index == len(filter_nums) - 1:
            c = c + temp
            
            
    #print('count', count)  
    #print('c', c) 
    
    return res, count, c
    
def prune_model(model_original, solution, args):
    model = copy.deepcopy(model_original)
    #newmodel = copy.deepcopy(model_original)
    
    #把一维矩阵根据卷积层改为二维矩阵
    cfg_mask, cfg, solution_new = masking(solution, args)
    print('cfg', cfg)
    #print('l', len(cfg))
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    
    newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
    if args.cuda:
        model.cpu()
        newmodel.cpu()
        
    conv_count = 1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            #第一层不剪枝
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask)))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask)))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask)))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    num_parameters1 = sum([param.nelement() for param in newmodel.parameters()])
    num_parameters2 = sum([param.nelement() for param in model.parameters()])
    #print('1', num_parameters1)
    #print('2', num_parameters2)
    #print(newmodel)
    #print('m', model)
    
        
    return newmodel, solution_new