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
    high = filter_nums[0]
    #print('solution', solution)
    residual_index = 1
    for index in range(len(filter_nums)):
        
        if index == 0:
            filters = solution[low : high]
            
        else:
            low = high
            high = filter_nums[index] + low
            filters = solution[low : high]#[0]
        
        temp = np.sum(filters)
        # count the mask for the first conv layer and the second conv layer in risidual block 
        # the last fully connected layer keeps unchanged
        if index % 3 != 0 and index < len(filter_nums) - 1:
            #print(index)
            res.append(filters)
            count.append(temp)
            

    #print('count', count)  
    #print('c', c) 
    
    return res, count
    
def prune_model(model_original, solution, args):
    model = copy.deepcopy(model_original)
    newmodel = copy.deepcopy(model_original)
    
    #把一维矩阵根据卷积层改为二维矩阵
    cfg_mask, cfg = masking(solution, args)
    #print('cfg', cfg)
    #print('l', len(cfg))
    
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    
    #newmodel = resnet(num_classes = 10)
    if args.cuda:
        model.cpu()
        newmodel.cpu()
        
    conv_count = 0
    #downsample_index = [15, 43, 79, 131]
    index = 0
    for [m0, m1] in zip(model.modules(), newmodel.modules()): 
        if isinstance(m0, nn.Conv2d):
            #print('conv_count', conv_count)
            #the first layer keeps unchaned
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                #print('first layer', m0.weight.data.size())
                conv_count += 1
                pre = m0.weight.data.clone()
                now = m0.weight.data.clone()
                continue
                
            now = m0.weight.data.clone()
            #print('before down now', now.size())
            #print('before down pre', pre.size())
            #the downsample layer
            if now.shape[1] != pre.shape[0]:
                m1.weight.data = m0.weight.data.clone()
                continue
            
            #pre = now.clone()
            
            #the first conv layer in residual block, we will prune filters(the fisrt size)
            if conv_count % 3 == 1:
                #print('1:')
                #print('mo', m0.weight.data.size())
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask)))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                #print('m1', w.size())
                layer_id_in_cfg += 1
                conv_count += 1
                continue
                
            #the second conv layer in residual block, we will prune filters and channels (the fisrt and second size)  
            elif conv_count % 3 == 2:
                #print('2:')
                #print('mo', m0.weight.data.size())
                mask1 = cfg_mask[layer_id_in_cfg]
                mask2 = cfg_mask[layer_id_in_cfg-1]
                idx1 = np.squeeze(np.argwhere(np.asarray(mask1)))
                idx2 = np.squeeze(np.argwhere(np.asarray(mask2)))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                if idx2.size == 1:
                    idx2 = np.resize(idx2, (1,))
                w = m0.weight.data[idx1.tolist(), :, :, :].clone()
                w = w[:, idx2.tolist(), :, :].clone()
                #print('m1', w.size())
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
                
            #the last conv layer in residual block, we will pruned channels (the second size)   
            elif conv_count % 3 == 0:
                #print('3:')
                #print('mo', m0.weight.data.size())
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask)))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                #print('m1', w.size())
                m1.weight.data = w.clone()
                conv_count += 1
                continue
                
        elif isinstance(m0, nn.BatchNorm2d): 
            residual_count = conv_count - 1
            #print('residual_count', residual_count)
            
            #the first layer
            if residual_count ==  0:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                pre = now.clone()
                continue
                
            else:   
                #the downsample layer
                if now.shape[1] != pre.shape[0]:
                    #print('downsample bn mo', m0.weight.data.size())
                    m1.weight.data = m0.weight.data.clone()
                    m1.bias.data = m0.bias.data.clone()
                    m1.running_mean = m0.running_mean.clone()
                    m1.running_var = m0.running_var.clone()
                    continue

                #the residual block   
                else:      
                    #the fisrt and second conv layer in residual block
                    if residual_count % 3 != 0:
                        #print('bn 1, 2 ,mo', m0.weight.data.size())
                        mask = cfg_mask[layer_id_in_cfg-1]
                        idx = np.squeeze(np.argwhere(np.asarray(mask)))
                        if idx.size == 1:
                            idx = np.resize(idx, (1,))
                        m1.weight.data = m0.weight.data[idx.tolist()].clone()
                        m1.bias.data = m0.bias.data[idx.tolist()].clone()
                        m1.running_mean = m0.running_mean[idx.tolist()].clone()
                        m1.running_var = m0.running_var[idx.tolist()].clone()
                        pre = now.clone()
                        continue
                    #the last conv layer in residual block and the first layer
                    #print('bn 3 mo', m0.weight.data.size())
                    m1.weight.data = m0.weight.data.clone()
                    m1.bias.data = m0.bias.data.clone()
                    m1.running_mean = m0.running_mean.clone()
                    m1.running_var = m0.running_var.clone()
                    pre = now.clone()
                
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            
        index += 1
        
    num_parameters1 = sum([param.nelement() for param in newmodel.parameters()])
    num_parameters2 = sum([param.nelement() for param in model.parameters()])
    #print('1', num_parameters1)
    #print('2', num_parameters2)
    #print(newmodel)
    #print('m', model)
    
      
    return newmodel