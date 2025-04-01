import argparse, os, logging, time
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch import nn
import torch
from torch import optim

from dataset_loader import query_loader

from xception import xception



from transform import TwoTransform, get_augs, get_augs1, MutilTransform

from tqdm import tqdm
import matplotlib.pyplot as plt

from torch import package
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from numpy import mean
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from torchmetrics import AUROC
from utils import test_epoch
from utils import train_CORE
import torchvision.transforms as transforms
import albumentations as A
import json
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import time
import random



class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)

def row_norms(X, squared=False):
	"""Row-wise (squared) Euclidean norm of X.
	Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
	matrices and does not create an X.shape-sized temporary.
	Performs no input validation.
	Parameters
	----------
	X : array_like
		The input array
	squared : bool, optional (default = False)
		If True, return squared norms.
	Returns
	-------
	array_like
		The row-wise (squared) Euclidean norm of X.
	"""
	norms = np.einsum('ij,ij->i', X, X)

	if not squared:
		np.sqrt(norms, norms)
	return norms

def outer_product_opt(c1, d1, c2, d2):
	"""Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products
	"""
	B1, B2 = c1.shape[0], c2.shape[0]
	t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)), np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
	t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)), np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
	t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
	t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
	t2 = t2.reshape(1, B2).repeat(B1, axis=0)
	return t1 + t2 - 2*t3

def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(1234), n_local_trials=None):
	"""Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)
	Parameters
	----------
	X1, X2 : array or sparse matrix
		The data to pick seeds for. To avoid memory copy, the input data
		should be double precision (dtype=np.float64).
	n_clusters : integer
		The number of seeds to choose
	init : list
		List of points already picked
	random_state : int, RandomState instance
		The generator used to initialize the centers. Use an int to make the
		randomness deterministic.
		See :term:`Glossary <random_state>`.
	n_local_trials : integer, optional
		The number of seeding trials for each center (except the first),
		of which the one reducing inertia the most is greedily chosen.
		Set to None to make the number of trials depend logarithmically
		on the number of seeds (2+log(k)); this is the default.
	Notes
	-----
	Selects initial cluster centers for k-mean clustering in a smart way
	to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
	"k-means++: the advantages of careful seeding". ACM-SIAM symposium
	on Discrete algorithms. 2007
	Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
	which is the implementation used in the aforementioned paper.
	"""

	n_samples, n_feat1 = X1.shape
	_, n_feat2 = X2.shape
	# x_squared_norms = row_norms(X, squared=True)
	centers1 = np.empty((n_clusters+len(init)-1, n_feat1), dtype=X1.dtype)
	centers2 = np.empty((n_clusters+len(init)-1, n_feat2), dtype=X1.dtype)

	idxs = np.empty((n_clusters+len(init)-1,), dtype=np.longlong)

	# Set the number of local seeding trials if none is given
	if n_local_trials is None:
		# This is what Arthur/Vassilvitskii tried, but did not report
		# specific results for other than mentioning in the conclusion
		# that it helped.
		n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = init

	centers1[:len(init)] = X1[center_id]
	centers2[:len(init)] = X2[center_id]
	idxs[:len(init)] = center_id

	# Initialize list of closest distances and calculate current potential
	distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init), -1)

	candidates_pot = distance_to_candidates.sum(axis=1)
	best_candidate = np.argmin(candidates_pot)
	current_pot = candidates_pot[best_candidate]
	closest_dist_sq = distance_to_candidates[best_candidate]

	# Pick the remaining n_clusters-1 points
	for c in range(len(init), len(init)+n_clusters-1):
		# Choose center candidates by sampling with probability proportional
		# to the squared distance to the closest existing center
		rand_vals = random_state.random_sample(n_local_trials) * current_pot
		candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
										rand_vals)
		# XXX: numerical imprecision can result in a candidate_id out of range
		np.clip(candidate_ids, None, closest_dist_sq.size - 1,
				out=candidate_ids)

		# Compute distances to center candidates
		distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(len(candidate_ids), -1)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates,
				   out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		idxs[c] = best_candidate

	return None, idxs[len(init)-1:]

def Random(model, dataset, args, device, num):
    data_info = {}
    with torch.no_grad():
        for idx, (data, label, path) in tqdm(enumerate(dataset),total=len(dataset)):
            data = data.to(device)
            if args.method != "CORE" and args.base_xception is False and args.method != "Con":
                _,_, output = model.encoder_q(data)[1]

            else:
                if args.mode == 1:
                    _,_, output = model(data)[1]
                else:
                    _, output = model(data)[1]
            # if isinstance(output, tuple):
            #     out_label = output[1]
            # else:
            #     out_label = output
            out_label = output

            out_label = out_label.cpu().numpy()
            labels = softmax(out_label)
            pred = np.max(labels,axis=1)

            for idx1, _ in enumerate(pred):

                data_info[path[idx1]] = {"label": int(label[idx1].cpu().numpy()), "pred": float(pred[idx1])}


    sort_info = sorted(data_info.items(), key=lambda item:item[1]["pred"])
    imgs = []
    for info in sort_info:
        imgs.append(info[0])
    random.shuffle(imgs)
    imgs_r = imgs[num:]
    imgs = imgs[:num]
    
    real_imgs = []
    fake_imgs = []
    real_imgs_r = []
    fake_imgs_r = []
    for img in imgs:
        if data_info[img]["label"] == 1:
            fake_imgs.append(img)
        else:
            real_imgs.append(img)
    if args.pre_labeled != "":
        with open(os.path.join(args.pre_labeled, "labeled_real.json")) as f3:
            real_imgs_t = json.load(f3)
            real_imgs = real_imgs + real_imgs_t
        with open(os.path.join(args.pre_labeled, "labeled_fake.json")) as f4:
            fake_imgs_t = json.load(f4)
            fake_imgs = fake_imgs + fake_imgs_t
    for img in imgs_r:
        if data_info[img]["label"] == 1:
            fake_imgs_r.append(img)
        else:
            real_imgs_r.append(img)

    if os.path.exists(args.output) is False:
        os.system("mkdir -p %s" % args.output)
    labeled_real_path = os.path.join(args.output, "labeled_real.json")
    labeled_fake_path = os.path.join(args.output, "labeled_fake.json")
    unlabeled_real_path = os.path.join(args.output, "unlabeled_real.json")
    unlabeled_fake_path = os.path.join(args.output, "unlabeled_fake.json")
    with open(labeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs))
    with open(labeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs))
    with open(unlabeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs_r))
    with open(unlabeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs_r))

def least_confidence(model, dataset, args, device, num):
    data_info = {}
    with torch.no_grad():
        for idx, (data, label, path) in tqdm(enumerate(dataset),total=len(dataset)):
            data = data.to(device)
            if args.method != "CORE" and args.base_xception is False and args.method != "Con":
                feat, output = model.encoder_q(data)

            else:
                feat, output = model(data)

            if args.mode == 1 or args.mode == 4:
                out_label = output[2]
            else:
                out_label = output[1]
            # if isinstance(output, tuple):
            #     out_label = output[1]
            # else:
            #     out_label = output
            # out_label = output
            out_label = out_label.cpu().numpy()
            labels = softmax(out_label)
            pred = np.max(labels,axis=1)

            for idx1, _ in enumerate(pred):

                data_info[path[idx1]] = {"label": int(label[idx1].cpu().numpy()), "pred": float(pred[idx1])}


    sort_info = sorted(data_info.items(), key=lambda item:item[1]["pred"])
    imgs = []
    for info in sort_info:
        imgs.append(info[0])
    imgs_r = imgs[num:]
    imgs = imgs[:num]
    
    real_imgs = []
    fake_imgs = []
    real_imgs_r = []
    fake_imgs_r = []
    for img in imgs:
        if data_info[img]["label"] == 1:
            fake_imgs.append(img)
        else:
            real_imgs.append(img)
    if args.pre_labeled != "":
        with open(os.path.join(args.pre_labeled, "labeled_real.json")) as f3:
            real_imgs_t = json.load(f3)
            real_imgs = real_imgs + real_imgs_t
        with open(os.path.join(args.pre_labeled, "labeled_fake.json")) as f4:
            fake_imgs_t = json.load(f4)
            fake_imgs = fake_imgs + fake_imgs_t
    for img in imgs_r:
        if data_info[img]["label"] == 1:
            fake_imgs_r.append(img)
        else:
            real_imgs_r.append(img)
    if os.path.exists(args.output) is False:
        os.system("mkdir -p %s" % args.output)
    labeled_real_path = os.path.join(args.output, "labeled_real.json")
    labeled_fake_path = os.path.join(args.output, "labeled_fake.json")
    unlabeled_real_path = os.path.join(args.output, "unlabeled_real.json")
    unlabeled_fake_path = os.path.join(args.output, "unlabeled_fake.json")
    with open(labeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs))
    with open(labeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs))
    with open(unlabeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs_r))
    with open(unlabeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs_r))

def BADGE(model, dataset, args, device, num):
    data_info = {}

    id2img = {}
    id2label = {}

    tgt_emb = torch.zeros([len(dataset.sampler), 2])
    tgt_pen_emb = torch.zeros([len(dataset.sampler), 2048])
    tgt_lab = torch.zeros(len(dataset.sampler))
    tgt_preds = torch.zeros(len(dataset.sampler))
    batch_sz = args.batch_size

    with torch.no_grad():
        for batch_idx, (data, label, path) in tqdm(enumerate(dataset),total=len(dataset)):
            for idx, p in enumerate(path):
                id2img[batch_idx*batch_sz+idx] = p
                id2label[batch_idx*batch_sz+idx] = int(label[idx].cpu().numpy())

            data = data.to(device)
            label = label.to(device)

            if args.method != "CORE" and args.base_xception is False and args.method != "Con":
                feat, output = model.encoder_q(data)

            else:
                feat, output = model(data)
            if args.mode == 1 or args.mode == 4:
                out_label = output[2]
            else:
                out_label = output[1]
            # if isinstance(output, tuple):
            #     out_label = output[1]
            # else:
            #     out_label = output

            tgt_pen_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feat.shape[0]), :] = feat.cpu()
            tgt_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, out_label.shape[0]), :] = out_label.cpu()
            tgt_lab[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, out_label.shape[0])] = label
            tgt_preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feat.shape[0])] = out_label.argmax(dim=1, keepdim=True).squeeze()


    # Compute uncertainty gradient
    tgt_scores = nn.Softmax(dim=1)(tgt_emb)
    tgt_scores_delta = torch.zeros_like(tgt_scores)
    tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1

    # Uncertainty embedding
    badge_uncertainty = (tgt_scores-tgt_scores_delta)
    # Seed with maximum uncertainty example
    max_norm = row_norms(badge_uncertainty.cpu().numpy()).argmax()

    _, q_idxs = kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), num, init=[max_norm])

    real_imgs = []
    fake_imgs = []
    real_imgs_r = []
    fake_imgs_r = []

    all_ids = [i for i in range(len(id2img))]
    r_ids = list(set(all_ids)-set(q_idxs))
    for idx in r_ids:
        if id2label[idx] == 1:
            fake_imgs_r.append(id2img[idx])
        else:
            real_imgs_r.append(id2img[idx])

    for idx in q_idxs:
        if id2label[idx] == 1:
            fake_imgs.append(id2img[idx])
        else:
            real_imgs.append(id2img[idx])
    if args.pre_labeled != "":
        with open(os.path.join(args.pre_labeled, "labeled_real.json")) as f3:
            real_imgs_t = json.load(f3)
            real_imgs = real_imgs + real_imgs_t
        with open(os.path.join(args.pre_labeled, "labeled_fake.json")) as f4:
            fake_imgs_t = json.load(f4)
            fake_imgs = fake_imgs + fake_imgs_t

    if os.path.exists(args.output) is False:
        os.system("mkdir -p %s" % args.output)
    labeled_real_path = os.path.join(args.output, "labeled_real.json")
    labeled_fake_path = os.path.join(args.output, "labeled_fake.json")
    unlabeled_real_path = os.path.join(args.output, "unlabeled_real.json")
    unlabeled_fake_path = os.path.join(args.output, "unlabeled_fake.json")
    with open(labeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs))
    with open(labeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs))
    with open(unlabeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs_r))
    with open(unlabeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs_r))

def CLUE(model, dataset, args, device, num):
    data_info = {}

    id2img = {}

    tgt_emb = torch.zeros([len(dataset.sampler), 2])
    if args.query_dim == "mode1":
        if args.mode == 1 or args.mode == 4:
            tgt_pen_emb = torch.zeros([len(dataset.sampler), 256])
        else:
            tgt_pen_emb = torch.zeros([len(dataset.sampler), 128])
    else:
        tgt_pen_emb = torch.zeros([len(dataset.sampler), 2048])
    tgt_lab = torch.zeros(len(dataset.sampler))
    tgt_preds = torch.zeros(len(dataset.sampler))
    batch_sz = args.batch_size

    with torch.no_grad():
        for batch_idx, (data, label, path) in tqdm(enumerate(dataset),total=len(dataset)):
            for idx, p in enumerate(path):
                id2img[batch_idx*batch_sz+idx] = p

            data = data.to(device)
            label = label.to(device)

            feat, output = model(data)

            if args.mode == 1 or args.mode == 4:
                out_label = output[2]
            else:
                out_label = output[1]
            if args.query_dim == "mode1":
                if args.mode == 1 or args.mode == 4:
                    feat = torch.cat((output[0], output[1]), 1)
                else:
                    feat = output[0]


            tgt_pen_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feat.shape[0]), :] = feat.cpu()
            tgt_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, out_label.shape[0]), :] = out_label.cpu()
            tgt_lab[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, out_label.shape[0])] = label
            tgt_preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, feat.shape[0])] = out_label.argmax(dim=1, keepdim=True).squeeze()
    
    T = args.T
    tgt_scores = nn.Softmax(dim=1)(tgt_emb / T)
    tgt_scores += 1e-8
    sample_weights = -(tgt_scores*torch.log(tgt_scores)).sum(1).cpu().numpy()
    st_time = time.time()
    # Run weighted K-means over embeddings
    km = KMeans(num)
    km.fit(tgt_pen_emb, sample_weight=sample_weights)

    # Find nearest neighbors to inferred centroids
    dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
    sort_idxs = dists.argsort(axis=1)
    q_idxs = []
    ax, rem = 0, num
    while rem > 0:
        q_idxs.extend(list(sort_idxs[:, ax][:rem]))
        q_idxs = list(set(q_idxs))
        rem = num-len(q_idxs)
        ax += 1
    ed_time = time.time()
    print(ed_time-st_time)
    imgs = []

    imgs_r = []

    all_ids = [i for i in range(len(id2img))]
    r_ids = list(set(all_ids)-set(q_idxs))
    for idx in r_ids:
        imgs_r.append(id2img[idx])

    for idx in q_idxs:
        imgs.append(id2img[idx])
    
    if os.path.exists(args.output) is False:
        os.system("mkdir -p %s" % args.output)
    unlabeled_path = os.path.join(args.output, "unlabeled.json")

    tolabeled_path = os.path.join(args.output, "tolabeled.json")

    with open(unlabeled_path, "w") as f:
        f.write(json.dumps(imgs_r))
    with open(tolabeled_path, "w") as f:
        f.write(json.dumps(imgs))


def MARGIN(model, dataset, args, device, num):
    data_info = {}
    with torch.no_grad():
        for idx, (data, label, path) in tqdm(enumerate(dataset),total=len(dataset)):
            data = data.to(device)
            if args.method != "CORE" and args.base_xception is False and args.method != "Con":
                _, output = model.encoder_q(data)
            else:
                _, output = model(data)
            # if isinstance(output, tuple):
            #     out_label = output[1]
            # else:
            #     out_label = output
            if args.mode == 1 or args.mode == 4:
                out_label = output[2]
            else:
                out_label = output[1]
            out_label = out_label.cpu().numpy()
            a1 = np.max(out_label, axis=1)
            a2 = np.min(out_label, axis=1)
            margin = a1 - a2

            for idx1, _ in enumerate(margin):

                data_info[path[idx1]] = {"label": int(label[idx1].cpu().numpy()), "pred": float(margin[idx1])}


    sort_info = sorted(data_info.items(), key=lambda item:item[1]["pred"])
    imgs = []
    for info in sort_info:
        imgs.append(info[0])
    imgs_r = imgs[num:]
    imgs = imgs[:num]
    real_imgs = []
    fake_imgs = []
    real_imgs_r = []
    fake_imgs_r = []
    for img in imgs:
        if data_info[img]["label"] == 1:
            fake_imgs.append(img)
        else:
            real_imgs.append(img)
    if args.pre_labeled != "":
        with open(os.path.join(args.pre_labeled, "labeled_real.json")) as f3:
            real_imgs_t = json.load(f3)
            real_imgs = real_imgs + real_imgs_t
        with open(os.path.join(args.pre_labeled, "labeled_fake.json")) as f4:
            fake_imgs_t = json.load(f4)
            fake_imgs = fake_imgs + fake_imgs_t
    for img in imgs_r:
        if data_info[img]["label"] == 1:
            fake_imgs_r.append(img)
        else:
            real_imgs_r.append(img)
    if os.path.exists(args.output) is False:
        os.system("mkdir -p %s" % args.output)
    labeled_real_path = os.path.join(args.output, "labeled_real.json")
    labeled_fake_path = os.path.join(args.output, "labeled_fake.json")
    unlabeled_real_path = os.path.join(args.output, "unlabeled_real.json")
    unlabeled_fake_path = os.path.join(args.output, "unlabeled_fake.json")
    with open(labeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs))
    with open(labeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs))
    with open(unlabeled_real_path, "w") as f:
        f.write(json.dumps(real_imgs_r))
    with open(unlabeled_fake_path, "w") as f:
        f.write(json.dumps(fake_imgs_r))



def main(args):
    print("args: {}".format(args))
    device = "cuda:0"
    # model
    if args.model_name == "xception":
        model = xception
    trans1 = get_augs(name="None",norm=args.norm)
    trans2 = get_augs(name="RE",norm=args.norm)
    trans3 = get_augs(name="RaAug",norm=args.norm)
    trans4 = get_augs(name="DFDC_selim",norm=args.norm)
    trans = [trans1,trans2,trans3,trans4]

    augs = MutilTransform(trans)


    test_augs = get_augs(name="None",norm=args.norm,size=args.size)
    print("test aug:{}".format(test_augs))

    # dataset
    train_dataset = query_loader(args.root, transform=test_augs)

    testloader = DataLoader(train_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last=False
    )

    # create model
    print("=> creating model '{}'".format(args.model_name))

    model = model(pretrained=True, num_classes=2, mode=args.mode)

    model = model.to(device)
    model.eval()

    if args.load_model_path:
        if os.path.isfile(args.load_model_path):
            print("=> loading checkpoint '{}'".format(args.load_model_path))
            if args.gpu is None:
                checkpoint = torch.load(args.load_model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.load_model_path, map_location=device)
            start_epoch = checkpoint["epoch"]

            if args.method == "CORE":
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["single_dict"])
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

    if args.query_type == "least_confidence":
        least_confidence(model, testloader, args, device, args.num)
        return
    elif args.query_type == "Random":
        Random(model, testloader, args, device, args.num)
        return
    elif args.query_type == "BADGE":
        BADGE(model, testloader, args, device, args.num)
        return
    elif args.query_type == "CLUE":
        CLUE(model, testloader, args, device, args.num)
        return
    elif args.query_type == "MARGIN":
        MARGIN(model, testloader, args, device, args.num)
        return





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--num-classes', type=int, default=2)
    arg('--method', type=str, default="")
    
    # consistency loss
    arg('--consistency', type=str, default="None",choices=["None","cos","L1","L2"])
    arg('--consistency-rate', type=float, default=1)

    # transforms
    arg('--aug-name', type=str, default="None",choices=["None","RE","RandCrop","RaAug","DFDC_selim"])
    arg('--norm', type=str, default="0.5",choices=["imagenet","0.5","0"])
    arg('--size', type=int, default=299)

    # dataset 
    arg('--dataset', type=str, default='ff')
    arg('--ff-quality', type=str, default='c23',choices=['c23','c40','raw'])
    arg('--root', type=str, default='/hy-tmp/data/sxt_data/FaceForensics++')
    arg('--batch-size', type=int, default=32)
    arg('--num-workers', type=int, default=8)
    arg('--shuffle', type=bool, default=True)
    arg('--test', type=bool, default=False)
    arg('--use_moco', type=bool, default=True)
    arg('--real-weight', type=float, default=4.0)

    # optimizer
    arg('--optimizer', type=str, default="adam")
    arg('--wd', '--weight-decay', type=float, default=1e-5)

    arg('--lr', type=float, default=0.001)

    arg('--exp-name', type=str, default='test')

    arg('--gpu', type=str, default=None)

    arg('--log-interval', type=int, default=100)

    arg("--epochs", type=int, default=50)
    arg("--load-model-path", type=str, default=None)
    arg("--output", type=str, default="./output")

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
        default=[10, 20],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )


    arg("--cos", action="store_true", help="use cosine lr schedule")
    arg("--fine_tune", action="store_true", help="use cosine lr schedule")
    arg("--query_type", default="ENT", type=str, help="query type")
    arg("--fake_base", type=str, nargs="+", default=['Deepfakes'])
    arg("--fake_type", type=str, nargs="+", default=['Deepfakes'])
    arg("--OOD", action="store_true", help="use OOD reg")
    arg("--save_info", action="store_true", help="save_info")
    arg("--from_file", default="", type=str)
    arg("--num", default=1958, type=int)
    arg("--base_xception", action="store_true", help="use base xception model")
    arg("--from_file1", action="store_true", help="")
    arg("--file_path", type=str, default="")
    arg("--f", type=float, default=0.05)
    arg("--pre_labeled", type=str, default="")
    arg("--query_dim", type=str, default="")
    arg("--mode", type=int, default=1)
    arg('--T', type=float, default=0.5)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed) 

    main(args)