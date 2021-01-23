#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
import argument
import os
# In[2]:

#返回每一层的的feature map，先用老方法，再用新方法做


def Merge(dict1, dict2):
    return(dict1.update(dict2))


def APoZ(compress_rate, model, args, train_loader, test_loader, val_loader):
    model.eval()
    val_loss = 0
    val_acc = 0
    test_loss = 0
    fc = dict()

    count = 0
    
    for img, lable in val_loader:
        lable = lable + 1
        if args.cuda:
            #64*1*28*28
            img = img.cuda()
            #64
            lable = lable.cuda()
            
        img = Variable(img)
        #64
        lable = Variable(lable)
        
  
        #print('forward')
        out, feature_map = model.forward(img)
        
        index = 0
        for x in feature_map:
            cell_value = x.data.cpu().numpy()
            if (str(index)) in fc:
                fc[str(index)] += np.array([np.sum(cell_value[:,j,:] <= 0) for j in range(x.shape[1])])
            else:
                fc[str(index)] = np.array([np.sum(cell_value[:,j,:] <= 0) for j in range(x.shape[1])])   
            index += 1
        
        loss = F.cross_entropy(out, lable)
        test_loss = loss.item() # sum up batch loss
        pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(lable.data.view_as(pred)).cpu().sum()
        
        
            
        if args.cuda:
            #args.test_batch_size = 1
            test_loss /= img.size(0)
            num = img.size(0)
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, num,
                100. * correct / num)) 
        
        else:  
            num = img.size(0)
            test_loss /= num
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, num,
                100. * correct / num))    
        #break
        model.reset()
        
    count_conv_fp = fc
    count_fc_fp = {'49': []}
                
    
    for (name, value) in count_conv_fp.items():
        #value = np.array(value)
        #print('name', name)
        #print('count_conv_fp[name]', count_conv_fp[name])
        num_t = int(compress_rate * value.shape[0])
        #print(num_t)
        #print(count_conv_fp.items)
        small_rank = np.argsort(value)[::-1]
        #print('rank value', value[small_rank])
        count_conv_fp[name] = small_rank[: num_t]
        #print('after count_conv_fp[name]', count_conv_fp[name])
        
        
    Merge(count_conv_fp, count_fc_fp)
    
    i = 0
    for (key, value) in count_conv_fp.items():

     

        if i % 3 == 0 or i == (len(args.filter_nums) - 1):
            #l = len(filters)
            count_conv_fp[key] = []

        i += 1
            
    return count_conv_fp
    


# In[14]:


def get_code(compress_rate, model_original, args, train_loader, test_loader, val_loader):
    model = copy.deepcopy(model_original)
    
    if args.cuda:
        model.cuda()
    #print(APoZ(conv_fp_dict, fc_fp_dict, count_conv_fp, count_fc_fp, num_t))
    return(APoZ(compress_rate, model, args, train_loader, test_loader, val_loader))
    #print(count_conv_fp, count_fc_fp)






