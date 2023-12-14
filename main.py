from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from tensorboardX import SummaryWriter
import models

from pytorchfi.core import FaultInjection as pfi_core
from fault_injection.my_weight_error_models import weight_bit_flip_func
from fault_injection.my_weight_error_models import *

import torch.nn.functional as F
import copy
import wandb
import time 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/bobzhou/dataset',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
    default='cifar10',
    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet20_quan',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='/home/bobzhou/BFA/save/2023-09-06/cifar10_resnet20_quan__SAM_QAT/model_best.pth.tar',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=1,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
parser.add_argument(
    '--optimize_step',
    dest='optimize_step',
    action='store_true',
    help='enable the step size optimization for weight quantization')
# Bit Flip Attacked
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=10,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
# load model trained in distribution
parser.add_argument(
    '--distributed',
    dest='distributed',
    action='store_true',
    help='loaded model trained in distribution.'
)
# binarization
parser.add_argument('--bin',
                    dest='bin',
                    action='store_true',
                    help='enable the binarization')
#  logging and visualization
parser.add_argument("--wandb_id", type=str)
# wandb visualization name
parser.add_argument('--wandb_name',
                    type=str,
                    default='SGD')

##########################################################################

args = parser.parse_args()




if not args.wandb_id:  #如果没有输入就重新生成
    args.wandb_id = wandb.util.generate_id()
# wandb.init(
#             project = "BFA",
#             config = args,
#             name = args.wandb_name,
#             # name = 'ResNet20_cifar10_SGD',
#             sync_tensorboard=True,
#             #resume = True,
#             )

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
# random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################


def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
        target_acc = 11
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
        target_acc = 11
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
        target_acc = 2
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
        target_acc = 11
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
        target_acc = 11
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
        target_acc = 0.2
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    # print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the parameters thus param groups can be updated by different optimizer
    # all_param = [
    #     param for name, param in net.named_parameters()
    #     if not 'step_size' in name
    # ]

    # step_param = [
    #     param for name, param in net.named_parameters() if 'step_size' in name
    # ]

    # if args.optimizer == "SGD":
    #     print("using SGD as optimizer")
    #     optimizer = torch.optim.SGD(all_param,
    #                                 lr=state['learning_rate'],
    #                                 momentum=state['momentum'],
    #                                 weight_decay=state['decay'],
    #                                 nesterov=True)

    # elif args.optimizer == "Adam":
    #     print("using Adam as optimizer")
    #     optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
    #                                         net.parameters()),
    #                                  lr=state['learning_rate'],
    #                                  weight_decay=state['decay'])

    # elif args.optimizer == "RMSprop":
    #     print("using RMSprop as optimizer")
    #     optimizer = torch.optim.RMSprop(
    #         filter(lambda param: param.requires_grad, net.parameters()),
    #         lr=state['learning_rate'],
    #         alpha=0.99,
    #         eps=1e-08,
    #         weight_decay=0,
    #         momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    # recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            # '''loading model'''
            # net=checkpoint['state_dict']
            # print(net)

            if args.distributed:
                # checkpoint = checkpoint['model_pos']
                new_checkpoint = {} ## 新建一个字典来访问模型的权值
                for k,value in checkpoint.items():
                    key = k.split('module.')[-1]
                    new_checkpoint[key] = value
                checkpoint = new_checkpoint
            # if not (args.fine_tune):
            #     args.start_epoch = checkpoint['epoch']
            #     recorder = checkpoint['recorder']
            #     optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                print(checkpoint['state_dict'])
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)
            net.load_state_dict(state_tmp)

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)


    # define fault injection model

    



    if args.enable_bfa:

        top1, top5, losses = validate(test_loader, net, criterion, log)
        print(f'clean accuracy:{top1}')
        
        pfi_model = weight_bit_flip_func(
                net,
                args.test_batch_size,
                input_shape=[3, 32, 32],
                layer_types=[torch.nn.Conv2d],
                use_cuda=args.use_cuda,
            )
        

        # times = 10
        # fault_rates = [1e-5, 1e-4,  1e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2]

        # for i in range(len(fault_rates)):
        #     fault_rate = fault_rates[i]
        #     avg = fault_injetion_test(fault_rate, times, pfi_model, test_loader, criterion, log)
        #     print('------------------------')
        #     print(f'fault_rate:{fault_rate}')
        #     print(f'avg:{avg}')



        avgs = []


        # times = 1
        # fault_rates = [1e-7, 1e-6, 1e-5]

        # for i in range(len(fault_rates)):
        #     fault_rate = fault_rates[i]
        #     avg = fault_injetion_test(fault_rate, times, pfi_model, test_loader, criterion, log)
        #     avgs.append(avg)
        #     print('------------------------')
        #     print(f'fault_rate:{fault_rate}')
        #     print(f'avg:{avg}')

        
        times = 50
        fault_rates = [ 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3,8e-3,9e-3,1e-2,2e-2, 3e-2,4e-2,5e-2]
        # fault_rates = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3,8e-3,9e-3]
        # fault_rates = [8e-4, 9e-4, 1e-3]

        for i in range(len(fault_rates)):
            
            fault_rate = fault_rates[i]
            print('------------------------')
            avg = fault_injetion_test(fault_rate, times, pfi_model, test_loader, criterion, log)
            avgs.append(avg)
            print(f'fault_rate:{fault_rate}')
            print(f'avg:{avg}')
        
        # times = 50
        # fault_rates = [7e-4, 8e-4, 9e-4, 1e-3]
        # for i in range(len(fault_rates)):
        #     fault_rate = fault_rates[i]
        #     avg = fault_injetion_test(fault_rate, times, pfi_model, test_loader, criterion, log)
        #     avgs.append(avg)
        #     print('------------------------')
        #     print(f'fault_rate:{fault_rate}')
        #     print(f'avg:{avg}')
        
        # times = 10
        # fault_rates = [3e-3, 4e-3, 5e-3, 1e-2]
        # fault_rates = [1e-2]
        # for i in range(len(fault_rates)):
        #     fault_rate = fault_rates[i]
        #     avg = fault_injetion_test(fault_rate, times, pfi_model, test_loader, criterion, log)
        #     avgs.append(avg)
        #     print('------------------------')
        #     print(f'fault_rate:{fault_rate}')
        #     print(f'avg:{avg}')



        # times = 1
        # fault_rates = [1e-7]

        # for i in range(len(fault_rates)):
        #     fault_rate = fault_rates[i]
        #     avg = fault_injetion_test(fault_rate, times, pfi_model, test_loader, criterion, log)
        #     avgs.append(avg)
        #     print('------------------------')
        #     print(f'fault_rate:{fault_rate}')
        #     print(f'avg:{avg}')

        print(f'avgs:{avgs}')

        return

    if args.evaluate:
        validate(test_loader, net, criterion, log)
        return

   

    log.close()



# 
def fault_injetion_test(fault_rate, times, pfi_model, test_loader, criterion, log):
    top1s = []
    for i in range(times):
        start_time = time.time()
        # print(f'fault_rate:{fault_rate}')
        print(f'the {i} fault injection')
        corrupt_model = multi_weight_inj_fault_rate(pfi_model, fault_rate=fault_rate)
        top1, top5, losses = validate(test_loader, corrupt_model, criterion, log)
        top1s.append(top1)
        end_time = time.time()
        print(f'top1:{top1}')
        print(f'time:{end_time-start_time}')

    print(top1s)
    sum = 0
    for i in range(len(top1s)):
        sum = sum + top1s[i]
    avg = sum/len(top1s)
    
    return avg


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(
                non_blocking=True
            )  # the copy will be non_blockinghronous with respect to the host.
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
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
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg



def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        # print_log(
        #     '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        #     .format(top1=top1, top5=top5, error1=100 - top1.avg), log)

    return top1.avg, top5.avg, losses.avg



def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()