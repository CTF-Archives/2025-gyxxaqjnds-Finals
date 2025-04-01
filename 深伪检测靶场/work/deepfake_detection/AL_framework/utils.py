import argparse, os, logging, time
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch import optim

from dataset_loader import train_loader

from xception import xception
from tqdm import tqdm
# from trainer import Trainer
import shutil
from transform import TwoTransform, get_augs, get_augs1
from consistency_loss import ConsistencyCos, ConsistencyL2, ConsistencyL1

import matplotlib.pyplot as plt

from torch import package
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from numpy import mean
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from torchmetrics import AUROC
import torch.nn.functional as F

weight_energy = torch.nn.Linear(2, 1).cuda()

logistic_regression = torch.nn.Linear(1, 2).cuda()

sample_number = 1000
sample_from = 10000
feature_dim = 2048

number_dict = {} 
data_dict = torch.zeros(2, sample_number, feature_dim).cuda() 
cls_real_list = [0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def cal_auc(pred,label):
    try:
        auc = roc_auc_score(label, pred)
        fprs, tprs, _ = roc_curve(label, pred)
        tdr = {}
        tdr["fpr"] = {}
        for t in [0.001,0.0001]:
            ind = 0
            for fpr in fprs:
                if fpr > t:
                    break
                ind += 1
            tdr[t] = tprs[ind-1]
            tdr["fpr"][t] = fprs[ind-1]
    except Exception as e:
        print(str(e))
        auc = 0
        tdr = {0.001:0,0.0001:0}
    return auc, tdr

def softmax(x,axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def evaluate1(output,label):
    output = softmax(output)

    pred_idx = np.argmax(output,axis=1)
    acc = np.sum(pred_idx==label)/len(label)
    auc, tdr = cal_auc(1-output[:,0],label)

    return acc, auc, tdr

def load_our_model(model_path):
    model_name = "CLBAL"
    if os.path.exists(model_path) is False:
        return None
    package_name = model_name
    resouce_name = model_name + ".pkl"
    imp = package.PackageImporter(model_path)
    loaded_model = imp.load_pickle(package_name, resouce_name)
    return loaded_model

def test_epoch(args):
    test_augs = get_augs(name="None",norm=args.norm,size=args.size)
    print("test aug:{}".format(test_augs))
    device = "cuda:0"
    # dataset
    test_dataset = train_loader(args.root, transform=test_augs,test=True)

    print("len test dataset:{}".format(len(test_dataset)))

    testloader = DataLoader(test_dataset,
        batch_size = 64,
        shuffle = True
    )

    # model
    if args.model_name == "xception":
        model = xception
    # create model
    print("=> creating model '{}'".format(args.model_name))

    model = model(pretrained=True, num_classes=args.num_classes, mode=args.mode)

    # optionally resume from a checkpoint
    if args.from_pt:
        print("++++++++++++++++++++")
        model = load_our_model("/home/user/build/deepmake-detection/src/app/deepfake_detection/models/CLBAL.pt")
        model = model.cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], find_unused_parameters=True)
        start_epoch = 0
    elif args.load_model_path:
        if os.path.isfile(args.load_model_path):
            print("=> loading checkpoint '{}'".format(args.load_model_path))
            if args.gpu is None:
                checkpoint = torch.load(args.load_model_path, map_location=torch.device('cuda:0'))
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.load_model_path, map_location=loc)
            start_epoch = checkpoint["epoch"]

            if args.method != "CORE":
                model.load_state_dict(checkpoint["single_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.load_model_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.load_model_path))
            start_epoch = 0

    model.eval()
    model.to(device)
    outputs = []
    labels = []

    with torch.no_grad():
        for batch_idx,(data,label,_) in tqdm(enumerate(testloader),total=len(testloader)):
            data = data.to(device)
            label = label.to(device)

            N = label.size(0)
            if args.base_xception:

                if args.mode == 0:
                    _, output = model(data)
                elif args.mode == 1:
                    _,_, output = model(data)[1]
                else:
                    _, output = model(data)[1]

                    # if isinstance(output, tuple):
                    #     output = output[1]

            output = output.data.cpu().numpy()
            print(output)

            label = label.data.cpu().numpy()
            outputs.append(output)
            labels.append(label)
    outputs = np.concatenate(outputs)
    labels = np.concatenate(labels)
    acc, auc, tdr = evaluate1(outputs,labels)
    print("{:.5f}, {:.5f}".format(acc, auc))



def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", save_dir=""):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir,"model_best.pth.tar"))