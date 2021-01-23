import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
import argument
import step1 
import step2 
import logging
import time
import models
import original_coding
#load data
args = argument.Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
logger = logging.getLogger(__name__)

basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(basic_format)    
log_path = 'depth%s_log.txt' % (args.depth)
print('log_path {}'.format(log_path))
handler = logging.FileHandler(log_path, 'a', 'utf-8')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('log_path {}'.format(log_path))


traindir = '../train'
valdir = '../val'
testdir = '../test'


gpu_id = "0,1,2,3" ; #指定gpu id
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id #这里的赋值必须是字符串，list会报错
device_ids = range(torch.cuda.device_count())  #torch.cuda.device_count()=2
#device_ids=[0,1] 这里的0 就是上述指定 2，是主gpu,  1就是7,模型和数据由主gpu分发


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.val_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.test_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)



def test(model):
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
        #print('input', input[0])
        print('target', target)
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
        #print('output', output[0])
        
        
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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
    print(' * Prec@5 {top5.avg:.3f}'
          .format(top5=top5))

    return top1.avg

def val(model):
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
    for i, (input, target) in enumerate(val_loader):
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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
    print(' * Prec@5 {top5.avg:.3f}'
          .format(top5=top5))

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
   
        
    model = models.__dict__[args.arch](pretrained=False, num_classes=1000)
    #save_path = '../logs/checkpoint%s.pth.tar' % (args.depth) 
    save_path = '../logs/checkpoint%s.pth.tar' % (args.depth) 
    if args.cuda:
        print('here')
        checkpoint = torch.load(save_path)
        newdict = dict()
        for name, para in checkpoint['state_dict'].items():
            name = name.replace('module.','')
            newdict[name] = para  


    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))   
        newdict = dict()
        for name, para in checkpoint['state_dict'].items():
            name = name.replace('module.','')
            newdict[name] = para  
                
        
    model.load_state_dict(newdict)
    
    return model

def construct_step2_model_load():
    model = models.__dict__[args.arch](pretrained=False, num_classes=1000)
    
    return model
        
        

model = construct_model_load()

retrain_model = construct_step2_model_load()

#for m in model.modules():
#    print('m', m)
'''   
for name, para in model.named_parameters():
    if para.requires_grad:
        
        if 'weight' in name and 'downsample' not in name:
            #如果是卷积层且参数维度为4
            if para.dim() == 4:
                print('name4:', name)
                print('para:', para.size())
                        

            elif para.dim() == 2:              
                
                print('name2:', name)
                print('para:', para.size())
        


if len(device_ids)>1:
    model=torch.nn.DataParallel(model);#前


for module in model.children():
    print('module', module)
'''    
model=torch.nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
'''
for i in range(1):
    test(model)
val(model)
'''
#test(model)
#val(model)
#get_code(model)

#original_coding.original_code(model, args, train_loader, test_loader, val_loader)

#step1.get_population(model, args, train_loader, test_loader, val_loader)

if args.retrain:
    step2.get_population(retrain_model, args, train_loader, test_loader, val_loader)
    
else:
    step2.get_population(model, args, train_loader, test_loader, val_loader)
