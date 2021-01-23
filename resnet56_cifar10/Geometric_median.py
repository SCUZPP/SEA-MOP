import torch
from torch.autograd import Variable
from torchvision import models
#import cv2
#import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import dataset
#from prune import *
#import argparse
from operator import itemgetter
from heapq import nsmallest
import time

from torchvision import datasets, transforms
import copy
import argument
from scipy.spatial import distance
import random



def GM_conv(conv_name, weight_torch, compress_rate, distance_rate, dist_type, args):
    #传进来的是每一个卷积层的filters的weight
    
    #计算出每一层需要剪枝的filters个数
    #print(weight_torch.size()[0])
    filter_pruned_num = int(weight_torch.size()[0] * compress_rate)
    similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
  
    
    #把每个卷积层展开成一行，有多少个filters得到多少行，正确
    weight_vec = weight_torch.view(weight_torch.size()[0], -1)
    
    #计算出每一个卷积核自己的L2范式，不是到其它卷积核的L2范式
    norm2 = torch.norm(weight_vec, 2, 1)
    norm2_np = norm2.cpu().numpy()
    
    filter_small_index = []
    filter_large_index = []
    
    #求出最大的卷积核
    filter_large_index = norm2_np.argsort()[filter_pruned_num : ]
    #print(filter_large_index)
    #求出最小的卷积核
    filter_small_index = norm2_np.argsort()[ : filter_pruned_num]

    #把卷积核的序号用64位整数存储
    if args.cuda:
        indices = torch.LongTensor(filter_large_index).cuda()
    else:
        indices = torch.LongTensor(filter_large_index)
    
    #只选出weight_vec中最大的几个卷积核存在weight_vec_after_norm中
    weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
    #print(weight_vec_after_norm)
    
    if dist_type == "l2" or "l1":
        #求出每一个卷积核到其它卷积核的欧式距离，每一行每一列存储的是到其它卷积核的距离
        similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
    
    #求出每个卷积核到其它卷积核距离的和
    similar_sum = np.sum(np.abs(similar_matrix), axis = 0)
    
    #for distance similar: get the filter index with largest similarity == small distance
    similar_large_index = similar_sum.argsort()[similar_pruned_num : ]
    similar_small_index = similar_sum.argsort()[ : similar_pruned_num]
    #print(similar_small_index)
 
    similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]
    #print(similar_index_for_filter)
    
    similar_index_for_filter = list(similar_index_for_filter)
    filter_small_index = list(filter_small_index)
    
    
    for i in similar_index_for_filter:
        filter_small_index.append(i)
        
    return filter_small_index
    
def get_code(compress_rate, model, args, train_loader, test_loader, val_loader):
    
    if args.cuda:
        model.cuda()
    
    res_GM = {}
    norm_rate = 0
    distance_rate = compress_rate 
    dist_type = "l2"
    index = 0

    #model.cpu()
    #model.named_parameters.cpu()

    #把每一个卷积层的filters weight传进函数中进行处理
    for name, para in model.named_parameters():
        if para.requires_grad:
            if 'weight' in name:
                if  para.dim() == 4:
                    img = para.data
                    #print(imag.shape)
                    
                    name = str(index)
                    res_GM[name] = GM_conv(name, img, norm_rate, distance_rate, dist_type, args)
                    index += 1
                    
    res_GM[str(index)] = [] 
    
    i = 0
    for (key, value) in res_GM.items():

        if i % 2 != 0:
            filters = value
            

        if i % 2 == 0 and i != 0:
            res_GM[key] = filters

        i += 1
        
                
    return res_GM           