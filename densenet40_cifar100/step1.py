import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy
import pickle
import random

from utils import *
from ec import *
from pruning import *
import os
    


def recode2binary(original_filter, integer_code, filter_nums):
    res = []
    temp = []
    for i in range(len(integer_code)):
        #index表示取组合中的哪一部分，取值范围为0-29
        index = i
        #value表示采用的是（0-199）种组合中的哪一种
        value = integer_code[i][0]
        #print(i)
        #print(value)
        temp = original_filter[value][index]
        #希望得到的是filters_length * 1
        for j in (temp):
            res.append(j)
    
    res = np.array(res)
    res = res.reshape(sum(filter_nums), 1)
    #print(res)
    #print(res.shape)
    #print(res)
    return res

def evaIndividual(ind, model, args, train_loader, test_loader, val_loader, original_filter):
    acc_30 = 0
    #gene_length = args.gene_length
    filter_nums = args.bn_nums
    gene_length = sum(np.array(filter_nums))
    
    solution = np.ones((sum(filter_nums), 1), dtype = np.int)
    integer_code = np.ones((gene_length, 1), dtype = np.int)
    #
    integer_code = ind.reshape(ind.shape[0], 1)
    #print(integer_code)
    #这行代码代表最后一个全连接层不进行剪枝，卷积核对应位始终是1，现在先不考虑
    temp = args.fc_nums
    temp = -temp
    solution[temp:] = 1  # Last 100 output should not be changed
    res = recode2binary(original_filter, integer_code, filter_nums)
    solution = res.reshape(res.shape[0], 1)
    
    #第一层和最后一层不剪枝
    #solution[-100:] = 1 
    #solution[-10:] = 1 
    # Prune model according to the solution
    model_new, solution_new = prune_model(model, solution, args)

    # Validate
    #只剪枝不微调
    acc, loss = test_forward(val_loader, model_new, args)
    
    #打印微调前的准确率和损失
    #old 
    pruning_rate_old = 1 - np.sum(solution) / (sum(filter_nums))
    #print('old', pruning_rate_old)
    #new
    pruning_rate = 1 - solution_new / (sum(filter_nums))
    #print('new', pruning_rate)
    
    print('step1微调前:  * accuracy {acc:.2f}, loss {loss:.2f}, pruning {pruning:.2f}'
         .format(acc = acc, loss = loss, pruning = pruning_rate))
    
    
    return 100-acc, solution_new, acc_30

class Individual():
    
    def __init__(self, gene_length, count, model, args, train_loader, test_loader, val_loader, original_filter):
        self.acc_30 = 0
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            self.dec[i] = count  # always begin with 1
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        self.evaluate(model, args, train_loader, test_loader, val_loader, original_filter) 

    def evaluate(self, model, args, train_loader, test_loader, val_loader, original_filter):
        self.obj[0], self.obj[1], self.acc_30 = evaIndividual(self.dec, model, args, train_loader, test_loader, val_loader, original_filter)
        
def initialization(pop_size, gene_length, model, args, train_loader, test_loader, val_loader, original_filter):
    population = []
    count = 0
    for i in range(pop_size):
        ind = Individual(gene_length, count, model, args, train_loader, test_loader, val_loader, original_filter)
        population.append(ind)
        count += 1
        
    return population
           


def get_population(model, args, train_loader, test_loader, val_loader):
    
    if args.depth == 40:  
        original_filter = pickle.load(open('filter_dict/filter_dict_densenet%s.pkl' % (args.depth), 'rb'))
            
        if not args.cuda:
            original_filter = original_filter[0 : 2]
            
        target_dir = 'Results_densenet40/'
        
    if args.cuda:
        model.cuda()


    # configuration
    #种群数量先设为8
    pop_size = len(original_filter)  # Population size
    n_obj = 2  # Objective variable dimensionality
    filter_nums = args.bn_nums
    dec_dim = args.gene_length  # Decision variable dimensionality
    
    if args.cuda:
        gen = 500 # Iteration number
        
    else:
        gen = 2

    p_crossover = 1  # crossover probability
    p_mutation = 1  # mutation probability

    # Initialization
    population = initialization(pop_size, dec_dim, model, args, train_loader, test_loader, val_loader, original_filter)

    g_begin = 0
    
    path_save = './' + target_dir
    
    for g in range(g_begin + 1, gen + 1):
        # generate reference lines and association
        #V, association, ideal = generate_ref_association(population)

        # Variation
        offspring = variation(population, p_crossover, p_mutation, model, args, train_loader, test_loader, val_loader, original_filter)

        # Update ideal point
        #PopObjs_Offspring = np.array([x.obj for x in offspring])
        #PopObjs_Offspring = np.vstack((ideal, PopObjs_Offspring))
        #ideal = np.min(PopObjs_Offspring, axis=0)

        # P+Q
        population.extend(offspring)

        # Environmental Selection
        population = environmental_selection(population, pop_size)

        # generation
        print('Gen:', g)

    #Save population
        if g == 1:
            with open(path_save + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population, f) 
    
        if g % 10 == 0:
            with open(path_save + "population-{}.pkl".format(g), 'wb') as f:
                pickle.dump(population, f) 
    
    print('step1 final population')

    population_init = []
    
    for ind in population:
        temp = ind.dec
        #temp = recode2binary(temp)
        population_init.append(temp)
        pruning_rate = 1 - ind.obj[1] / (sum(filter_nums)) 
        print(' * accuracy {acc:.2f}, pruning {pruning:.2f}'
             .format(acc = 100 - ind.obj[0], pruning=pruning_rate))
     
    #以文件形式存储
    pickle.dump(population_init, open('population_dict/population_dict_densenet%s.pkl' % (args.depth), 'wb'))


 