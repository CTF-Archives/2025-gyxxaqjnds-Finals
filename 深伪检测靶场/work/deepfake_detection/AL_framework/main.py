import sys
import argparse, os, logging, time
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch import optim

from dataset_loader import train_loader

import shutil
from xception import xception
import torch.nn.functional as F

# from trainer import Trainer

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
from utils import test_epoch

from losses import SupConLoss

# from GPUtil import showUtilization as gpu_usage

torch.multiprocessing.set_sharing_strategy("file_system")

def load_our_model(model_path):
    model_name = "CLBAL"
    if os.path.exists(model_path) is False:
        return None
    package_name = model_name
    resouce_name = model_name + ".pkl"
    imp = package.PackageImporter(model_path)
    loaded_model = imp.load_pickle(package_name, resouce_name)
    return loaded_model

def Model_Package(args):
    model = xception
    model = model(num_classes=args.num_classes, mode=args.mode)
    model = model.cuda()

    if args.load_model_path:
        if os.path.isfile(args.load_model_path):
            print("=> loading checkpoint '{}'".format(args.load_model_path))
            if args.gpu is None:
                checkpoint = torch.load(args.load_model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.load_model_path, map_location=loc)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["single_dict"])

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.load_model_path, checkpoint["epoch"]
                )
            )
    package_model(model)

def package_model(model):

    resouce_name = "CORE.pkl"
    package_name = "CORE"
    # resouce_name = os.path.join("./packge_model", resouce_name)
    with package.PackageExporter("./%s.pt" % package_name) as exp:

        # exp.intern("xception.**")
        exp.save_pickle(package_name, resouce_name, model)


def main(args):
    print(args)
    if args.test:
        test_epoch(args)
        return
    # os.environ['CUDA_VISIBLE_DEVICES'] = [2,3] 
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.nprocs = ngpus_per_node
    print(args)
    print(ngpus_per_node)
    if args.method == "CORE":
        train_CORE(0, 1, args)
        return
    if ngpus_per_node >= 1:
        if args.method == "CORE":
            main_worker = train_CORE
        else:
            main_worker = main_worker1
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,args))
    else:
        main_worker(model, args.gpu, ngpus_per_node, args, trainloader, testloader, save_dir)


def main_worker1(gpu, ngpus_per_node, args):
    args.rank = gpu
    args.batch_size = int(args.batch_size / ngpus_per_node)
    dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=ngpus_per_node, rank=args.rank)
    torch.cuda.set_device(gpu)
    
    save_dir = os.path.join("/tmp", args.exp_name,args.model_name)
    # save_dir = "/hy-tmp/data/sxt_data/exp_two_stage_AL/model_df_finetune_random/"
    os.makedirs(save_dir, exist_ok=True)

    logfile = '{}/{}.log'.format(save_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger = logging.getLogger()

    print("args: {}".format(args))

    # model
    if args.model_name == "xception":
        model = xception

    # transforms
    train_augs = get_augs(name=args.aug_name,norm=args.norm,size=args.size)
    consistency_fn=None
    if args.consistency != "None" and args.base_xception is False:
        train_augs = TwoTransform(train_augs)
        consistency_fn = ConsistencyCos()
    elif args.supcon:
        print("using supcon")
        train_augs = TwoTransform(train_augs)
        consistency_fn = SupConLoss()

    test_augs = get_augs(name="None",norm=args.norm,size=args.size)

    # dataset
    train_dataset = train_loader(args.root, transform=train_augs)
    test_dataset = train_loader(args.root, transform=test_augs, test=True)

    print("len train dataset:{}".format(len(train_dataset)))
    print("len test dataset:{}".format(len(test_dataset)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    # dataloader
    trainloader = DataLoader(train_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )
    testloader = DataLoader(test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        sampler=test_sampler,
        pin_memory=True
    )

    # create model
    print("=> creating model '{}'".format(args.model_name))

    model = model(num_classes=args.num_classes, mode=args.mode)

    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    loss_fn = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        weight_decay=args.wd
    )

    # optionally resume from a checkpoint
    if args.from_pt:
        model = load_our_model("/home/user/build/deepmake-detection/src/app/deepfake_detection/models/CLBAL.pt")
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        start_epoch = 0
    elif args.load_model_path:
        if os.path.isfile(args.load_model_path):
            print("=> loading checkpoint '{}'".format(args.load_model_path))
            if args.gpu is None:
                checkpoint = torch.load(args.load_model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.load_model_path, map_location=loc)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            # if args.train_basexception is False:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.load_model_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.load_model_path))
            start_epoch = 0
    else:
        start_epoch = 0
    if args.AL:
        start_epoch = 0
    

    # cudnn.benchmark = True
    best_auc = 0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.supcon:
            train_supcon(trainloader, model, consistency_fn, loss_fn, optimizer, epoch, args)
        elif args.base_xception:
            train_basexception(trainloader, model, criterion, loss_fn, optimizer, epoch, args)

        acc, auc = test_basexception(testloader, epoch, model, args)

        if args.rank == 0:
            print("[{}/epoch_{}][acc:{:.3f}][auc:{:.3f}]".format(save_dir,epoch,acc,auc))
            is_best=False
            if acc > best_auc:
                is_best=True
                best_auc = acc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "acc": acc,
                    "auc": auc,
                    "state_dict": model.state_dict(),
                    "single_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                filename=os.path.join(save_dir,"checkpoint_{:04d}.pth.tar".format(epoch)),
                save_dir=save_dir
            )

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def evaluate(output,label):
    auroc = AUROC(task="binary")
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output)

    # pred_idx = np.argmax(output,axis=1)
    _, pred_idx = output.topk(1, 1, True, True)
    pred_idx = pred_idx.squeeze(dim=1)
    acc = pred_idx.eq(label).sum() / label.size(0)
    try:
        auc = auroc(output[:,1], label)
    except Exception as e:
        pass
    return acc, auc

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def train(train_loader, model, criterion, loss_fn, optimizer, epoch, args, consistency_fn):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    train_acc = AverageMeter("Acc", ":6.4f")
    train_auc = AverageMeter("Auc", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, train_acc, train_auc],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (images, label) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        if args.fine_tune is False and args.only_CL is False:
            # if args.gpu is not None:
            images[0] = images[0].cuda(args.rank, non_blocking=True)
            images[1] = images[1].cuda(args.rank, non_blocking=True)

            label = label.cuda(args.rank, non_blocking=True)
            

            q, output, target = model(im_q=images[0], im_k=images[1])


            loss_consistency = criterion(output, target)
            loss_ce = loss_fn(q,label)
            # print(q)
            if epoch > 5:
                loss = args.consistency_rate * loss_consistency + loss_ce
            else:
                loss = (args.consistency_rate/5) * loss_consistency + loss_ce
            



        else:
            images = torch.cat(images,dim=0)
            label = torch.cat([label,label],dim=0)

            images = images.cuda(args.rank, non_blocking=True)
            label = label.cuda(args.rank, non_blocking=True)
            feature, _,  q = model(images)[1]

            loss_ce = loss_fn(q,label)

            loss_consistency = consistency_fn(feature)
            loss = args.consistency_rate * loss_consistency + loss_ce


        # q = q.data.cpu().numpy()
        # label = label.data.cpu().numpy()
        N = label.size(0)
        if args.only_CL is False:
            acc, auc = evaluate(q,label)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        if args.only_CL is False:
            reduced_acc = reduce_mean(acc, args.nprocs)
            reduced_auc = reduce_mean(auc, args.nprocs)
            train_acc.update(reduced_acc.item(), N)
            train_auc.update(reduced_auc.item(), N)
        losses.update(reduced_loss.item(), N)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and args.rank == 0:
            progress.display(i)

def train_basexception(train_loader, model, criterion, loss_fn, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    train_acc = AverageMeter("Acc", ":6.4f")
    train_auc = AverageMeter("Auc", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, train_acc, train_auc],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, label, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
            # if args.gpu is not None:
        images = images.cuda(args.rank, non_blocking=True)

        label = label.cuda(args.rank, non_blocking=True)

        # compute output
        if args.mode == 0:
            _, q = model(images)
        elif args.mode == 1:
            _,_, q = model(images)[1]
        else:
            _, q = model(images)[1]

            # if isinstance(q, tuple):
            #     q = q[1]

        loss_ce = loss_fn(q,label)
        loss = loss_ce

        # q = q.data.cpu().numpy()
        # label = label.data.cpu().numpy()
        N = label.size(0)

        acc, auc = evaluate(q,label)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)

        reduced_acc = reduce_mean(acc, args.nprocs)
        reduced_auc = reduce_mean(auc, args.nprocs)
        train_acc.update(reduced_acc.item(), N)
        train_auc.update(reduced_auc.item(), N)
        losses.update(reduced_loss.item(), N)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and args.rank == 0:
            progress.display(i)

def train_supcon(train_loader, model, consistency_fn, loss_fn, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    train_acc = AverageMeter("Acc", ":6.4f")
    train_auc = AverageMeter("Auc", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, train_acc, train_auc],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, label, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
            # if args.gpu is not None:

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda(args.rank, non_blocking=True)
        labels = label.detach()

        label = torch.cat([label,label],dim=0)

        label = label.cuda(args.rank, non_blocking=True)
        labels = labels.cuda(args.rank, non_blocking=True)
        bsz = labels.shape[0]
        # compute output
        if args.mode == 0:
            _, q = model(images)
        elif args.mode == 1:
            features,_, q = model(images)[1]
        else:
            features, q = model(images)[1]

            # if isinstance(q, tuple):
            #     q = q[1]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_con = consistency_fn(features, labels)

        loss_ce = loss_fn(q,label)
        loss = loss_ce + args.consistency_rate * loss_con

        # q = q.data.cpu().numpy()
        # label = label.data.cpu().numpy()
        N = label.size(0)

        acc, auc = evaluate(q,label)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)

        reduced_acc = reduce_mean(acc, args.nprocs)
        reduced_auc = reduce_mean(auc, args.nprocs)
        train_acc.update(reduced_acc.item(), N)
        train_auc.update(reduced_auc.item(), N)
        losses.update(reduced_loss.item(), N)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and args.rank == 0:
            progress.display(i)



def test_basexception(test_loader, epoch, model, args):
    model.eval()

    accs = []
    aucs = []
    with torch.no_grad():

        for i, (images, label, _) in enumerate(test_loader):

            images = images.cuda(args.rank, non_blocking=True)
            label = label.cuda(args.rank, non_blocking=True)
            N = label.size(0)


            if args.mode == 0:
                _, q = model(images)
            elif args.mode == 1:
                _,_, q = model(images)[1]
            else:
                _, q = model(images)[1]


            acc, auc = evaluate(q,label)
            torch.distributed.barrier()
            reduced_acc = reduce_mean(acc, args.nprocs)
            reduced_auc = reduce_mean(auc, args.nprocs)
            accs.append(reduced_acc.item())
            aucs.append(reduced_auc.item())

    return mean(accs), mean(aucs)

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", save_dir=""):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir,"model_best.pth.tar"))

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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--num-classes', type=int, default=2)
    arg('--method', type=str, default="")
    
    # consistency loss
    arg('--consistency', type=str, default="None",choices=["None","cos","L1","L2"])
    arg('--consistency-rate', type=float, default=0.2)

    # transforms
    arg('--aug-name', type=str, default="None",choices=["None","RE","RandCrop","RaAug","DFDC_selim"])
    arg('--norm', type=str, default="0.5",choices=["imagenet","0.5","0"])
    arg('--size', type=int, default=299)

    # dataset 
    arg('--dataset', type=str, default='ff')
    arg('--ff-quality', type=str, default='c23',choices=['c23','c40','raw'])
    arg('--root', type=str, default='/hy-tmp/data/sxt_data/FaceForensics++/')
    arg('--batch-size', type=int, default=32)
    arg('--num-workers', type=int, default=8)
    arg('--shuffle', type=bool, default=True)
    arg('--test', type=bool, default=False)
    arg('--use_moco', type=bool, default=True)
    arg('--real-weight', type=float, default=4.0)

    # optimizer
    arg('--optimizer', type=str, default="adam")
    arg('--wd', '--weight-decay', type=float, default=1e-5)

    arg('--lr', type=float, default=0.0002)

    arg('--exp-name', type=str, default='test')

    arg('--gpu', type=str, default=None)

    arg('--log-interval', type=int, default=100)

    arg("--epochs", type=int, default=50)
    arg("--load-model-path", type=str, default=None)

    arg("--model-name", type=str, default="xception",choices=["xception"])

    arg("--amp", default=False, action='store_true')

    arg("--seed", type=int, default=3407) # https://arxiv.org/abs/2109.08203

    arg(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )

    arg(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )

    arg(
        "--schedule",
        default=[50, 50],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )


    arg("--cos", action="store_true", help="use cosine lr schedule")

    arg("--num_query", type=int, default=100)


    arg("--AL", action="store_true", help="do AL experiment")
    arg("--f", type=float, default=0.05)
    arg("--base_xception", action="store_true", help="use base xception model")


    arg("--mode1", action="store_true", help="")

    arg("--supcon", action="store_true", help="")
    arg("--mode", type=int, default=1)
    arg("--package_model", action="store_true", help="")
    arg("--from_pt", action="store_true", help="")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed) 
    if args.package_model:
        Model_Package(args)
        sys.exit(0)
    main(args)
