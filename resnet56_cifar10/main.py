import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
import argument
#from vgg import *
import original_coding 
import step1 
import step2 
import logging
import models
import time
from cmp import *

#load data
args = argument.Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
logger = logging.getLogger(__name__)

basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(basic_format)    
log_path = 'depth%s_rand_%s_log.txt' % (args.depth, args.rand)
print('log_path {}'.format(log_path))
handler = logging.FileHandler(log_path, 'a', 'utf-8')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('log_path {}'.format(log_path))
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_set = datasets.CIFAR10('data.cifar10', train=True, download=True, 
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

if args.cuda:
    train_set = train_set
    #valset, train_set = torch.utils.data.random_split(train_set, [5000, 45000])
    
else:
    train_set, rand = torch.utils.data.random_split(train_set, [1000, 49000])
    valset, train_set = torch.utils.data.random_split(train_set, [500, 500])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)


testset = datasets.CIFAR10('data.cifar10', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ]))

if args.cuda:
    
    valset = testset
    
    
else:
    testset, rand= torch.utils.data.random_split(testset, [100, 9900])
    #testset = testset


#valset = testset
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=True, **kwargs)  
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

def test():
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        if args.cpu == False:
            input = input.cuda()#async=True)
            target = target.cuda()#async=True)

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Computes the precision@k for the specified values of k"""
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

#加载模型和数据
def construct_model_load():
   
    if args.depth == 56:
        
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
        if args.cuda:
            checkpoint = torch.load('logs/checkpoint.pth.tar')
            newdict = dict()
            for name, para in checkpoint['state_dict'].items():
                name = name.replace('module.','')
                newdict[name] = para  


        else:
            checkpoint = torch.load('logs/checkpoint.pth.tar', map_location=torch.device('cpu'))   
            newdict = dict()
            for name, para in checkpoint['state_dict'].items():
                name = name.replace('module.','')
                newdict[name] = para  
                
    
    model.load_state_dict(newdict)
    
    return model

def construct_step2_model_load():

    if args.depth == 56:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    
    return model
        

model = construct_model_load()
retrain_model = construct_step2_model_load()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#test()

#print_model_param_flops(model)

#original_coding.original_code(model, args, train_loader, test_loader, val_loader)

#step1.get_population(model, args, train_loader, test_loader, val_loader)

if args.retrain:
    step2.get_population(retrain_model, args, train_loader, test_loader, val_loader)
    
else:
    step2.get_population(model, args, train_loader, test_loader, val_loader)
    

