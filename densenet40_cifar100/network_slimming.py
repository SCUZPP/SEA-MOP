import copy
import torch.nn as nn
import torch
def get_code(compress_rate, model_original, args, train_loader, test_loader, val_loader):
    
    model = copy.deepcopy(model_original)
    #print('model', model)
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
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    #thre_index = int(total * compress_rate)
    thre_index = int(total * compress_rate)
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
            cfg_mask.append(mask.clone().cpu())
            #print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            #    format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
            
    res = []
    
    for mask in cfg_mask:
        temp = []
        for m in mask:
            temp.append(int(m))
        res.append(temp)
    
    #the fc layer
    temp = [1] * 100
    res.append(temp)
    pruned_ratio = pruned/total
    #newmodel = densenet(dataset=args.dataset, depth=args.depth, cfg=cfg)
    #print('cfg', cfg)
    return res

  
    