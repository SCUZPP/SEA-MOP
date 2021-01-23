#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
from math import *
#from scipy.spatial.distance import cdist

'''
# Domination check
def dominate(p, q):
    result = False
    for i, j in zip(p.obj, q.obj):
        if i < j:  # at least less in one dimension
            result = True
        elif i > j:  # not greater in any dimension, return false immediately
            return False
    return result

'''
def dominate(p,q):
    result = False
    #if p is feasible and q is infeasible
    if p.obj[0] <= 50 and q.obj[0] > 50:
        return True
    
    #if p in infeasible and q is feasible
    if p.obj[0] > 50 and q.obj[0] <= 50:
        return False
    
    #if p is feasible and q is feasible
    if p.obj[0] <= 50 and q.obj[0] <= 50:
        for i, j in zip(p.obj, q.obj):
            if i < j:  # at least less in one dimension
                result = True
                
            elif i > j:  # not greater in any dimension, return false immediately
                return False
            
    #if p is infeasible and q is infeasible
    if p.obj[0] > 50 and q.obj[0] > 50:
        if p.obj[0] > q.obj[0]:
            return False
        
        elif p.obj[0] < q.obj[0]:
            return True
        
        else:
            if p.obj[1] < q.obj[1]:
                return True
            
            else:
                return False
            
    return result 


def non_dominate_sorting(population):
    # find non-dominated sorted
    dominated_set = {}
    dominating_num = {}
    rank = {}
    for p in population:
        dominated_set[p] = []
        dominating_num[p] = 0

    sorted_pop = [[]]
    rank_init = 0
    for i, p in enumerate(population):
        for q in population[i + 1:]:
            if dominate(p, q):
                dominated_set[p].append(q)
                dominating_num[q] += 1
            elif dominate(q, p):
                dominating_num[p] += 1
                dominated_set[q].append(p)
        # rank 0
        if dominating_num[p] == 0:
            rank[p] = rank_init # rank set to 0
            sorted_pop[0].append(p)

    while len(sorted_pop[rank_init]) > 0:
        current_front = []
        for ppp in sorted_pop[rank_init]:
            for qqq in dominated_set[ppp]:
                dominating_num[qqq] -= 1
                if dominating_num[qqq] == 0:
                    rank[qqq] = rank_init + 1
                    current_front.append(qqq)
        rank_init += 1

        sorted_pop.append(current_front)

    return sorted_pop


class Individual():
    def __init__(self, gene_length, count, model, args, train_loader, test_loader, val_loader, original_filter):
        self.acc_30 = 0
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            self.dec[i] = count  # random binary code
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        self.evaluate(model, args, train_loader, test_loader, val_loader, original_filter)

    def evaluate(self, model, args, train_loader, test_loader, val_loader, original_filter):
        # self.obj[0], self.obj[1] = evaCNN(self.dec)
        print('test evaluate ec')
        self.obj[0], self.obj[1] = 0, 0


def initialization(pop_size, gene_length, model, args, train_loader, test_loader, val_loader, original_filter):
    population = []
    count = 0
    for i in range(pop_size):
        ind = Individual(gene_length, count, model, args, train_loader, test_loader, val_loader, original_filter)
        population.append(ind)
        count += 1
    return population


def evaluation(population, model, args, train_loader, test_loader, val_loader, original_filter):
    # Evaluation
    [ind.evaluate(model, args, train_loader, test_loader, val_loader, original_filter) for ind in population]
    return population


# one point crossover
#只需要改交叉和变异的编码，不需要改变啊，调用评价函数时会自动解码
#需要改变交叉和变异的方法，现在基因位数是30，每一位是0-199之间的整数
def one_point_crossover(p, q):
    gene_length = len(p.dec)
    child1 = np.zeros(gene_length, dtype=np.uint8)
    child2 = np.zeros(gene_length, dtype=np.uint8)
    k = np.random.randint(gene_length)
    child1[:k] = p.dec[:k]
    child1[k:] = q.dec[k:]

    child2[:k] = q.dec[:k]
    child2[k:] = p.dec[k:]

    return child1, child2


# Bit wise mutation
#对基因每一位以一定概率变异，变异操作是选取（0-199）中不等于原数值的数替换
def bitwise_mutation(p, p_m, size):
    gene_length = len(p.dec)
    population_size = size
    p_mutation = p_m / gene_length
    p_mutation = 0.01  ## constant mutation rate
    for i in range(gene_length):
        if np.random.random() < p_mutation:
            k = np.random.randint(population_size)
            while k == p.dec[i]:
                k = np.random.randint(population_size)
            p.dec[i] = k
    return p


# Variation (Crossover & Mutation)
def variation(population, p_crossover, p_mutation, model, args, train_loader, test_loader, val_loader, original_filter):
    offspring = copy.deepcopy(population)
    len_pop = int(np.ceil(len(population) / 2) * 2) 
    candidate_idx = np.random.permutation(len_pop)
    population_size = len(population)

    # Crossover
    for i in range(int(len_pop/2)):
        if np.random.random()<=p_crossover:
            individual1 = offspring[candidate_idx[i]]
            individual2 = offspring[candidate_idx[-i-1]]
            [child1, child2] = one_point_crossover(individual1, individual2)
            offspring[candidate_idx[i]].dec[:] = child1
            offspring[candidate_idx[-i-1]].dec[:] = child2

    # Mutation
    for i in range(len_pop):
        individual = offspring[i]
        offspring[i] = bitwise_mutation(individual, p_mutation, population_size)

    # Evaluate offspring
    offspring = evaluation(offspring, model, args, train_loader, test_loader, val_loader, original_filter)

    return offspring


# Crowding distance
def crowding_dist(population):
    pop_size = len(population)
    crowding_dis = np.zeros((pop_size,))

    obj_dim_size = len(population[0].obj)
    # crowding distance
    for m in range(obj_dim_size):
        obj_current = [x.obj[m] for x in population]
        sorted_idx = np.argsort(obj_current)  # sort current dim with ascending order
        obj_max = np.max(obj_current)
        obj_min = np.min(obj_current)

        # keep boundary point
        crowding_dis[sorted_idx[0]] = np.inf
        crowding_dis[sorted_idx[-1]] = np.inf
        for i in range(1, pop_size - 1):
            crowding_dis[sorted_idx[i]] = crowding_dis[sorted_idx[i]] + \
                                                      1.0 * (obj_current[sorted_idx[i + 1]] - \
                                                             obj_current[sorted_idx[i - 1]]) / (obj_max - obj_min)
            
    return crowding_dis




def environmental_selection(population, n):
    pop_sorted = non_dominate_sorting(population)
    selected = []
    for front in pop_sorted:
        if len(selected) < n:
            if len(selected) + len(front) <= n:
                selected.extend(front)
            else:
                # select individuals according crowding distance here
                crowding_dst = crowding_dist(front)
                k = n - len(selected)
                dist_idx = np.argsort(crowding_dst, axis=0)[::-1]  # descending order, large rank small angel
                for i in dist_idx[:k]:
                    selected.extend([front[i]])
                break
    return selected