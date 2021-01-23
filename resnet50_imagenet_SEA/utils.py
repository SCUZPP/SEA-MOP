import math

import torch.nn.init as init
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import pickle

from scipy.spatial.distance import cdist
from scipy.special import comb
import itertools
import time
import logging
logger = logging.getLogger("__main__")
#加载数据


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class FilterPrunner:
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)    # 32x32

        x = self.model.layer1(x)  # 32x32
        x = self.model.layer2(x)  # 16x16
        x = self.model.layer3(x)  # 8x8

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        return x



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#finetuning

#微调模型
#epoch设为几呢？10
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target + 1
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(train_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            

def validate(val_loader, model, criterion, args):
    #print('val be')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target + 1
        #print('val af')
        if args.cuda:
            input = input.cuda()#async=True)
            target = target.cuda()#async=True)
            model.cuda()
            criterion.cuda()

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    


    return top1.avg, losses.avg

def test(test_loader, model, criterion, args):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target + 1
        if args.cuda:
            input = input.cuda()#async=True)
            target = target.cuda()#async=True)
            model.cuda()

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()
        '''
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        '''
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
            
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
 

    return top1.avg, losses.avg
   
#validation用的model到底是不是微调之后的model？还是什么model？
#每一次epoch之后的model的参数是否发生了变化
#训练数据，对数据进行backward微调，再进行validation，返回acc, loss
def train_forward(train_loader, test_loader, val_loader, model, args):
    #只加载一次
    if args.cuda:
        model.cuda()
        
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_loss = 1000
    acc_30 = []
    
    if args.retrain:
        epochs = args.re_epochs
    else:
        epochs = args.ft_epochs
        
    for epoch in range(0, epochs):
        if args.retrain:
            optimizer = optim.SGD(model.parameters(), lr = args.retrain_lr, momentum = args.momentum, weight_decay = args.weight_decay)
            adjust_learning_rate(optimizer, epoch, args)

        else:
            optimizer = optim.SGD(model.parameters(), lr = args.ft_lr, momentum = args.momentum, weight_decay = args.weight_decay)

        train(train_loader, model, criterion, optimizer, epoch, args)
        #test(epoch)
        acc, loss = test(test_loader, model, criterion, args)

        if acc > best_acc:
            best_acc = acc
            best_loss = loss

        if epoch == 49 or epoch == 99 or epoch == 149 or epoch == 199 or epoch == 249 or epoch ==299:
            acc_30.append(best_acc)

                
            
    #acc, loss = validate(val_loader, model, pruner)
    return best_acc, best_loss, acc_30

def test_forward(val_loader, model, args):  # Validate
    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()    
  
    acc, loss = validate(val_loader, model, criterion, args)
    return acc, loss



