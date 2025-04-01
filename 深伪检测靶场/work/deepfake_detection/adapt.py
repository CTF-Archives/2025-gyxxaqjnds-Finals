import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import copy
import numpy as np



class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


class PseudoLabel(Algorithm):
    def __init__(self, num_classes, num_domains, config, model):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(num_classes, num_domains, config)
        self.model = model
        self.optimizer = choose_optimizer(model, config)
        self.beta = 0.8
        self.steps = 1
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, inference=True, adapt=True):
        if adapt:
            for _ in range(self.steps):
                self.model.eval()
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                self.model.train()
        else:
            outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        py, y_prime = F.softmax(outputs['cls'], dim=-1).max(1)

        flag = py > self.beta

        loss = F.cross_entropy(outputs['cls'][flag], y_prime[flag])
        loss.backward()
        optimizer.step()
        return outputs

    def predict(self, x, adapt=False):
        return self(x, adapt)

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class T3A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, num_classes, num_domains, config, model):
        super().__init__(num_classes, num_domains, config)
        # self.featurizer = algorithm.featurizer
        # self.classifier = algorithm.classifier
        self.model = model
        warmup_supports = self.model.head.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.model.head(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = -1  # 1 5 20 50 100  -1
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=True, inference=True):
        z = self.model.features(x)

        if adapt:
            # online adaptation
            p = self.model.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))

        pred = z @ torch.nn.functional.normalize(weights, dim=0)
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {'cls': pred, 'prob': prob, 'feat': z}

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)

        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data



class BatchEnsemble(nn.Module):
    def __init__(self, indim, outdim, ensemble_size, init_mode):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.in_features = indim
        self.out_features = outdim

        # register parameters
        self.register_parameter(
            "weight", nn.Parameter(
                torch.Tensor(self.out_features, self.in_features)
            )
        )
        self.register_parameter(
            "bias", nn.Parameter(
                torch.Tensor(self.out_features)
            )
        )

        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_features)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_features)
            )
        )

        use_ensemble_bias = True
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_features)
                )
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        self.init_mode = init_mode
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1 = x.size()
        r_x = x.unsqueeze(0).expand(self.ensemble_size, B, D1) #
        r_x = r_x.view(self.ensemble_size, -1, D1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x

    def reset(self):
        init_details = [0,1]
        initialize_tensor(self.weight, self.init_mode, init_details)
        initialize_tensor(self.alpha_be, self.init_mode, init_details)
        initialize_tensor(self.gamma_be, self.init_mode, init_details)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")
        if self.bias is not None:
            initialize_tensor(self.bias, "zeros")

def initialize_tensor(
        tensor: torch.Tensor,
        initializer: str,
        init_values: list = [],
    ) -> None:

    if initializer == "zeros":
        nn.init.zeros_(tensor)

    elif initializer == "ones":
        nn.init.ones_(tensor)

    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])

    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])

    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0 * init_values[1] * torch.bernoulli(
                    torch.zeros_like(tensor) + init_values[0]
                ) - init_values[1]
            )
    elif initializer == 'xavier_normal':
        torch.nn.init.xavier_normal_(tensor)

    elif initializer == 'kaiming_normal':
        torch.nn.init.kaiming_normal_(tensor)
    else:
        raise NotImplementedError(
            f"Unknown initializer: {initializer}"
        )

class TAST(Algorithm):
    def __init__(self, num_classes, num_domains, config, model):
        super().__init__(num_classes, num_domains, model)
        # trained feature extractor and last linear classifier
        # self.featurizer = algorithm.featurizer
        # self.classifier = algorithm.classifier

        # store supports and corresponding labels
        self.model = model
        warmup_supports = self.model.head.weight.data
        # warmup_supports = self.model.backbone.last_linear.weight.data
        # warmup_supports = self.model.backbone.last_layer.weight.data
        # warmup_supports = self.model.backbone_rgb.last_linear.weight.data  # srm
        self.warmup_supports = warmup_supports
        # warmup_prob = self.model.head(self.warmup_supports)
        warmup_prob = self.model.classifier(self.warmup_supports)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.ent = self.warmup_ent.data

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data

        # hparams
        self.filter_K = -1
        self.steps = 1
        self.num_ensemble = 5
        self.lr = 0.00001
        self.tau = 1
        self.init_mode = "kaiming_normal"
        self.num_classes = num_classes
        self.k = 1  # 1 2 4 8
        feat_dim= 768

        # multiple projection heads and its optimizer
        self.mlps = BatchEnsemble(feat_dim, feat_dim // 4, self.num_ensemble,
                                  self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)

    def forward(self, x, adapt=True, inference=True):
        z = self.model.features(x)

        if adapt:
            p_supports = self.model.classifier(z)
            yhat = torch.nn.functional.one_hot(p_supports.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p_supports)


            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])


        supports, labels = self.select_supports()

        for _ in range(self.steps):
            p = self.forward_and_adapt(z, supports, labels)
        prob = torch.softmax(p, dim=1)[:, 1]
        return {'cls': p, 'prob': prob, 'feat': z}

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)
        mlp_supports = mlp(supports)

        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
        for ens in range(self.num_ensemble):
            temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ mlp_supports[
                                                                                    ens * N: (ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B: (ens + 1) * B], dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)


            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]

        return logits

    def select_supports(self):
        '''
        we filter support examples with high prediction entropy
        :return: filtered support examples.
        '''
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]


        return self.supports, self.labels

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels):
        # targets : pseudo labels, outputs: for prediction
        with torch.no_grad():
            targets, outputs = self.target_generation(z, supports, labels)

        self.optimizer.zero_grad()

        loss = None
        logits = self.compute_logits(z, supports, labels, self.mlps)

        for ens in range(self.num_ensemble):
            if loss is None:
                loss = F.kl_div(logits[ens].log_softmax(-1), targets[ens])
            else:
                loss += F.kl_div(logits[ens].log_softmax(-1), targets[ens])

        loss.backward()
        self.optimizer.step()

        return outputs  # outputs

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.mlps.reset()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)

        torch.cuda.empty_cache()

    # from https://jejjohnson.github.io/research_journal/snippets/numpy/euclidean/
    def euclidean_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def target_generation(self, z, supports, labels):
        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)

        W = torch.exp(-dist)  # [B, N]

        temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes

        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else

        temp_labels = self.compute_logits(supports, supports, labels, self.mlps)  # [ens, N, C]
        temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]

        topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        #targets = targets.mean(0)  # [B,C]

        # outputs for prediction
        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]

        return targets, outputs


class TAST_v2(Algorithm):
    def __init__(self, num_classes, num_domains, config, model):
        super().__init__(num_classes, num_domains, model)
        # trained feature extractor and last linear classifier
        # self.featurizer = algorithm.featurizer
        # self.classifier = algorithm.classifier

        # store supports and corresponding labels
        self.model = model
        warmup_supports = self.model.head.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.model.head(self.warmup_supports)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.ent = self.warmup_ent.data

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data

        # hparams
        self.filter_K = -1
        self.steps = 1
        self.num_ensemble = 5
        self.lr = 0.00001
        self.tau = 0.2
        self.init_mode = "kaiming_normal"
        self.num_classes = num_classes
        self.k = 16  # 1 2 4 8

        # multiple projection heads and its optimizer
        self.mlps = BatchEnsemble(768, 768 // 2, self.num_ensemble,
                                  self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)

    def forward(self, x, adapt=True, inference=True):
        z = self.model.features(x)

        if adapt:
            p_supports = self.model.classifier(z)
            yhat = torch.nn.functional.one_hot(p_supports.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p_supports)


            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])


        supports, labels = self.select_supports()

        for _ in range(self.steps):
            p = self.forward_and_adapt(z, supports, labels)
        prob = torch.softmax(p, dim=1)[:, 1]
        return {'cls': p, 'prob': prob, 'feat': z}

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)
        mlp_supports = mlp(supports)

        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
        for ens in range(self.num_ensemble):
            temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ mlp_supports[
                                                                                    ens * N: (ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B: (ens + 1) * B], dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)


            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]

        return logits

    def select_supports(self):
        '''
        we filter support examples with high prediction entropy
        :return: filtered support examples.
        '''
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]


        return self.supports, self.labels

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels):
        # targets : pseudo labels, outputs: for prediction
        with torch.no_grad():
            targets, outputs = self.target_generation(z, supports, labels)

        self.optimizer.zero_grad()

        loss = None
        logits = self.compute_logits(z, supports, labels, self.mlps)

        for ens in range(self.num_ensemble):
            if loss is None:
                loss = F.kl_div(logits[ens].log_softmax(-1), targets[ens])
            else:
                loss += F.kl_div(logits[ens].log_softmax(-1), targets[ens])

        loss.backward()
        self.optimizer.step()

        return outputs  # outputs

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.mlps.reset()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)

        torch.cuda.empty_cache()

    # from https://jejjohnson.github.io/research_journal/snippets/numpy/euclidean/
    def euclidean_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def distance2(self, X, Y):
        XX1, X1 = torch.sort(X, descending=True, dim=-1)
        YY1, Y1 = torch.sort(Y, descending=True, dim=-1)

        return self.cosine_distance_einsum(XX1[:,:], YY1[:,:])

        # B = X.size(0)
        # N = Y.size(0)
        # W = torch.zeros([B,N]).to(X.device)
        # for i in range(B):
        #     for j in range(N):
        #         W[i,j] = (X1[i]==Y1[j]).sum()

        # return -W

    def target_generation(self, z, supports, labels):
        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)

        W = torch.exp(-dist)  # [B, N]

        temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes

        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else

        temp_labels = self.compute_logits(supports, supports, labels, self.mlps)  # [ens, N, C]
        temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]

        topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        #targets = targets.mean(0)  # [B,C]

        # outputs for prediction
        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]

        return targets, outputs


class T3A_v2(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, num_classes, num_domains, config, model):
        super().__init__(num_classes, num_domains, config)
        # self.featurizer = algorithm.featurizer
        # self.classifier = algorithm.classifier
        self.model = model
        warmup_supports = self.model.head.weight.data  # clip
        # warmup_supports = self.model.backbone.last_linear.weight.data  # xception
        # warmup_supports = self.model.backbone.last_layer.weight.data  # effnb4
        # warmup_supports = self.model.backbone_rgb.last_linear.weight.data  # srm
        self.warmup_supports = warmup_supports

        warmup_prob = self.model.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = 1000  # 1 5 20 50 100  -1
        self.steps=3
        self.num_ensemble = 5
        self.lr = 0.000005
        self.tau = 0.2
        self.beta = 0.7
        self.k = 1
        self.init_mode = "kaiming_normal"
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)
        feat_dim= 768

        # multiple projection heads and its optimizer
        self.mlps = BatchEnsemble(feat_dim, feat_dim // 2, self.num_ensemble,
                                  self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)
        self.consistency_loss = nn.MSELoss()


    def forward(self, x, adapt=True, inference=True):
        z = self.model.features(x)
        # p = self.model.classifier(z)
        # prob = torch.softmax(p, dim=1)[:, 1]
        # return {'cls': p, 'prob': prob, 'feat': z}
        if adapt:
            # online adaptation
            p = self.model.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        for _ in range(self.steps):
            # self.model.eval()
            p = self.forward_and_adapt(z, supports, labels, p)
        prob = torch.softmax(p, dim=1)[:, 1]
        return {'cls': p, 'prob': prob, 'feat': z}

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels, p):
        # targets : pseudo labels, outputs: for prediction
        with torch.no_grad():
            py = F.softmax(p, dim=-1)
            py1, _ = F.softmax(p, dim=-1).max(1)
            flag = py1 > self.beta
            targets, outputs = self.target_generation(z, supports, labels)


        self.optimizer.zero_grad()

        loss = None
        logits = self.compute_logits(z, supports, labels, self.mlps)


        for ens in range(self.num_ensemble):
            if loss is None:
                loss =  F.kl_div(logits[ens].log_softmax(-1), targets[ens]) + 10 * F.kl_div(logits[ens][flag].log_softmax(-1), py[flag]) 
            else:
                loss +=  F.kl_div(logits[ens].log_softmax(-1), targets[ens]) + 10 * F.kl_div(logits[ens][flag].log_softmax(-1), py[flag]) 

        loss.backward()
        self.optimizer.step()

        outputs1 = logits.mean(0).unsqueeze(1)

        outputs = outputs.unsqueeze(1)
        outputs = torch.concat((outputs1, outputs), 1)
        outputs = outputs.sum(1)
        outputs = outputs / (outputs.sum(1, keepdim=True) + 1e-12)  # [ens, B, C]


        return outputs  # outputs

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def target_generation(self, z, supports, labels):
        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)

        W = torch.exp(-dist)  # [B, N]

        temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes

        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else

        temp_labels = self.compute_logits(supports, supports, labels, self.mlps)  # [ens, N, C]
        temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]

        topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        # temp_feat = torch.bmm(topk_indices, temp_feat)  # [ens, B, dim]

        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        # temp_feat = temp_feat / (temp_feat.sum(2, keepdim=True) + 1e-12)
        #targets = targets.mean(0)  # [B,C]

        # outputs for prediction
        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]
        # outputs = outputs / (outputs.sum(1, keepdim=True) + 1e-12)  # [ens, B, C]
       

        return targets, outputs

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)
        mlp_supports = mlp(supports)

        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
        # feat = torch.zeros(self.num_ensemble, B, dim // 4).to(z.device)
        for ens in range(self.num_ensemble):
            temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ mlp_supports[
                                                                                    ens * N: (ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B: (ens + 1) * B], dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)


            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]
            # feat[ens] = temp_z

        return logits #, feat

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)

        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def target_generation(self, z, supports, labels):
        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)

        W = torch.exp(-dist)  # [B, N]

        temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes

        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else

        temp_labels = self.compute_logits(supports, supports, labels, self.mlps)  # [ens, N, C]
        temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]

        topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        # temp_feat = torch.bmm(topk_indices, temp_feat)  # [ens, B, dim]

        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        # temp_feat = temp_feat / (temp_feat.sum(2, keepdim=True) + 1e-12)
        #targets = targets.mean(0)  # [B,C]

        # outputs for prediction
        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]
        # outputs = outputs / (outputs.sum(1, keepdim=True) + 1e-12)  # [ens, B, C]
       

        return targets, outputs

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)
        mlp_supports = mlp(supports)

        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
        # feat = torch.zeros(self.num_ensemble, B, 768 // 2).to(z.device)
        for ens in range(self.num_ensemble):
            temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ mlp_supports[
                                                                                    ens * N: (ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B: (ens + 1) * B], dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)

            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]
            # feat[ens] = temp_z

        return logits

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)

        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


class T3A_v3(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, num_classes, num_domains, config, model):
        super().__init__(num_classes, num_domains, config)
        # self.featurizer = algorithm.featurizer
        # self.classifier = algorithm.classifier
        self.model = model
        warmup_supports = self.model.head.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.model.head(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = 1000  # 1 5 20 50 100  -1
        self.steps=1
        self.num_ensemble = 5
        self.lr = 0.00001
        self.tau = 0.2
        self.beta = 0.7
        self.k = 16
        self.init_mode = "kaiming_normal"
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

        # multiple projection heads and its optimizer
        self.mlps = BatchEnsemble(768, 768//2, self.num_ensemble,
                                  self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)

        self.set_requires_grad(self.model, False)

    def forward(self, x, adapt=True, inference=True):
        z = self.model.features(x)
        # p = self.model.classifier(z)
        # prob = torch.softmax(p, dim=1)[:, 1]
        # return {'cls': p, 'prob': prob, 'feat': z}

        if adapt:
            # online adaptation
            p = self.model.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        for _ in range(self.steps):
            p = self.forward_and_adapt(z, supports, labels, p)
        prob = torch.softmax(p, dim=1)[:, 1]
        return {'cls': p, 'prob': prob, 'feat': z}

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels, p):
        # targets : pseudo labels, outputs: for prediction
        # with torch.no_grad():
        #     py, _ = F.softmax(p, dim=-1).max(1)
        #     targets = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        #     flag = py > self.beta
            # targets = self.target_generation(z, supports, labels)  # 深为检测领域，pseudo labels信息相对不准确，利用最近邻预测信息

        self.optimizer.zero_grad()

        loss = None
    
        # with torch.no_grad():
        logits = self.compute_logits(z, supports, labels, self.mlps)

        for ens in range(self.num_ensemble):
            py, y_prime = F.softmax(logits[ens], dim=-1).max(1)

            flag = py > self.beta
            if loss is None:
                loss = F.cross_entropy(logits[ens][flag], y_prime[flag]) #F.kl_div(logits[ens][flag].log_softmax(-1), y_prime[flag])
            else:
                loss += F.cross_entropy(logits[ens][flag], y_prime[flag]) #F.kl_div(logits[ens][flag].log_softmax(-1), y_prime[flag])

        loss.backward()
        self.optimizer.step()

        outputs = logits.mean(0)

        return outputs  # outputs

    def target_generation(self, z, supports, labels):
        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)

        W = torch.exp(-dist)  # [B, N]

        temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes

        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else

        B = supports.size(0)
        supports_mlp = self.mlps(supports)

        temp_labels = self.compute_logits(supports_mlp, B)  # [ens, N, C]
        temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]

        topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        #targets = targets.mean(0)  # [B,C]

        # outputs for prediction
        # outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        # outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        # outputs = outputs.mean(0)  # [B,C]

        return targets

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def set_requires_grad(self, model, val):
        for p in model.parameters():
            p.requires_grad = val

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = z
        mlp_supports = supports

        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
        # feat = torch.zeros(self.num_ensemble, B, dim // 4).to(z.device)
        for ens in range(self.num_ensemble):
            temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ mlp_supports[
                                                                                    ens * N: (ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B: (ens + 1) * B], dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)


            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]
            # feat[ens] = temp_z

        return logits #, feat

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)

        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data



class T3A_wo_A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, num_classes, num_domains, config, model):
        super().__init__(num_classes, num_domains, config)
        # self.featurizer = algorithm.featurizer
        # self.classifier = algorithm.classifier
        self.model = model
        warmup_supports = self.model.head.weight.data
        # warmup_supports = self.model.backbone.last_linear.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.model.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = -1  # 1 5 20 50 100  -1
        self.steps=1
        self.num_ensemble = 5
        self.lr = 0.00001
        self.tau = 0.2
        self.beta = 0.7
        self.k = 16
        self.init_mode = "kaiming_normal"
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)
        feat_dim= 768

        # multiple projection heads and its optimizer
        self.mlps = BatchEnsemble(feat_dim, feat_dim // 2, self.num_ensemble,
                                  self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)
        self.consistency_loss = nn.MSELoss()


    def forward(self, x, adapt=True, inference=True):
        z = self.model.features(x)
        # p = self.model.classifier(z)
        # prob = torch.softmax(p, dim=1)[:, 1]
        # return {'cls': p, 'prob': prob, 'feat': z}
        if adapt:
            # online adaptation
            p = self.model.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        for _ in range(self.steps):
            # self.model.eval()
            p = self.forward_and_adapt(z, supports, labels, p)
        prob = torch.softmax(p, dim=1)[:, 1]
        return {'cls': p, 'prob': prob, 'feat': z}

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels, p):
        # targets : pseudo labels, outputs: for prediction
        with torch.no_grad():
            py = F.softmax(p, dim=-1)
            py1, _ = F.softmax(p, dim=-1).max(1)
            flag = py1 > self.beta
            targets, outputs = self.target_generation(z, supports, labels)


        self.optimizer.zero_grad()

        loss = None
        logits = self.compute_logits(z, supports, labels, self.mlps)


        for ens in range(self.num_ensemble):
            if loss is None:
                # print(feat[ens].shape, temp_feat[ens].shape)
                loss =  F.kl_div(logits[ens].log_softmax(-1), targets[ens]) + 10 * F.kl_div(logits[ens][flag].log_softmax(-1), py[flag]) #* self.consistency_loss(feat[ens], temp_feat[ens])
            else:
                loss +=  F.kl_div(logits[ens].log_softmax(-1), targets[ens]) + 10 * F.kl_div(logits[ens][flag].log_softmax(-1), py[flag]) #* self.consistency_loss(feat[ens], temp_feat[ens])

        loss.backward()
        self.optimizer.step()

        outputs1 = logits.mean(0).unsqueeze(1)

        outputs = outputs.unsqueeze(1)
        outputs = torch.concat((outputs1, outputs), 1)
        outputs = outputs.sum(1)
        outputs = outputs / (outputs.sum(1, keepdim=True) + 1e-12)  # [ens, B, C]

        # print(outputs1.shape, outputs.shape)

        return outputs  # outputs

