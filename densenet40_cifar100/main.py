import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
import argument
import original_coding 
import step1 
import step2 
import logging
from models import *
import time


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

train_set = datasets.CIFAR100('../data.cifar100', train=True, download=True, 
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))


if args.cuda:
    
    train_set = train_set
    valset, train_set = torch.utils.data.random_split(train_set, [1000, 49000])

    
else:
    train_set, rand = torch.utils.data.random_split(train_set, [1000, 49000])
    valset, train_set = torch.utils.data.random_split(train_set, [500, 500])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)


testset = datasets.CIFAR100('../data.cifar100', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ]))

if args.cuda:
    
    testset = testset
    #valset = testset
    #testset, valset= torch.utils.data.random_split(testset, [5000, 5000])
    
else:
    testset, rand= torch.utils.data.random_split(testset, [100, 9900])
    #testset = testset


#valset = testset
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=True, **kwargs)  
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)



def test(model):
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
   
    if args.depth == 40:
        
        model = densenet(depth=args.depth, dataset=args.dataset)
        if args.cuda:
            checkpoint = torch.load('../logs/checkpoint.pth.tar')
            newdict = dict()
            for name, para in checkpoint['state_dict'].items():
                name = name.replace('module.','')
                newdict[name] = para  


        else:
            checkpoint = torch.load('../logs/checkpoint.pth.tar', map_location=torch.device('cpu'))   
            newdict = dict()
            for name, para in checkpoint['state_dict'].items():
                name = name.replace('module.','')
                newdict[name] = para  
                
    
    model.load_state_dict(newdict)
    
    return model

def construct_step2_model_load():

    if args.depth == 40:
        model = densenet(depth=args.depth, dataset=args.dataset)
    
    return model
        
def get_code(model):
    
    model = copy.deepcopy(model)
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print('old',num_parameters)
    
    if args.cuda:
        model.cuda()
        
    #acc = test(model)
    
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            print('bn', m.weight.data.shape)
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * 0.1)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    if args.cuda:
        model.cuda()
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total
    #newmodel = densenet(dataset=args.dataset, depth=args.depth, cfg=cfg)
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print('t', num_parameters)
    print('cfg', cfg)
    print('pruned_ratio', pruned_ratio)

    print('Pre-processing Successful!')

    # simple test model after Pre-processing prune (simple set BN scales to zeros)


    #acc = test(model)

    print("Cfg:")
    #print(cfg_mask)

    newmodel = densenet(dataset=args.dataset, depth=args.depth, cfg=cfg)
    #print('oldmodel', model)
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    print('middle',num_parameters)

    if args.cuda:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
 
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
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
                continue

        elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()


    #print(newmodel)
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    print('new',num_parameters)
    model = newmodel
    #print('newmodel', newmodel)
    #acc = test(model)
    

model = construct_model_load()

retrain_model = construct_step2_model_load()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

#test(model)
#get_code(model)

#original_coding.original_code(model, args, train_loader, test_loader, val_loader)

#step1.get_population(model, args, train_loader, test_loader, val_loader)

if args.retrain:
    step2.get_population(retrain_model, args, train_loader, test_loader, val_loader)
    
else:
    step2.get_population(model, args, train_loader, test_loader, val_loader)
    

