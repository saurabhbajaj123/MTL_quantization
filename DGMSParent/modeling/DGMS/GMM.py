# Copyright (c) Runpei Dong, ArChip Lab.

""" DGMS GM Sub-distribution implementation.

Author: Runpei Dong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import DGMSParent.config as cfg

from DGMSParent.utils.misc import cluster_weights

class GaussianMixtureModel(nn.Module):
    """Concrete GMM for sub-distribution approximation.
    """
    def __init__(self, num_components, init_weights, temperature=0.01, init_method="k-means"):
        super(GaussianMixtureModel, self).__init__()
        self.num_components = num_components
        self.temperature = temperature
        print("self.num_components = {}, self.temperature = {}".format(self.num_components, self.temperature))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(self.device)
        self.params_initialization(init_weights, init_method)

    def params_initialization(self, init_weights, method='k-means'):
        """ Initialization of GMM parameters using k-means algorithm. """
        print(""" Initialization of GMM parameters using k-means algorithm. """)
        self.mu_zero = torch.tensor([0.0], device=self.device).float()
        self.pi_k, self.mu, self.sigma = \
                torch.ones(self.num_components-1, device=self.device), \
                torch.ones(self.num_components-1, device=self.device), \
                torch.ones(self.num_components-1, device=self.device)
        if method == 'k-means':
            initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
        elif method == 'empirical':
            initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            sigma_init, _sigma_zero = torch.ones_like(sigma_init).mul(0.01).cuda(), torch.ones_like(torch.tensor([_sigma_zero])).mul(0.01).cuda()
        self.mu = nn.Parameter(data=torch.mul(self.mu.cuda(), initial_region_saliency.flatten().cuda()))
        self.pi_k = nn.Parameter(data=torch.mul(self.pi_k.cuda(), pi_init)).cuda().float()
        self.pi_zero = nn.Parameter(data=torch.tensor([pi_zero_init], device=self.device)).cuda().float()
        self.sigma_zero = nn.Parameter(data=torch.tensor([_sigma_zero], device=self.device)).float()
        self.sigma = nn.Parameter(data=torch.mul(self.sigma, sigma_init)).cuda().float()
        self.temperature = nn.Parameter(data=torch.tensor([self.temperature], device=self.device))

    def gaussian_mixing_regularization(self):
        pi_tmp = torch.cat([self.pi_zero, self.pi_k], dim=-1).abs()
        return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).cuda()

    def Normal_pdf(self, x, _pi, mu, sigma):
        """ Standard Normal Distribution PDF. """
        return torch.mul(torch.reciprocal(torch.sqrt(torch.mul( \
               torch.tensor([2 * math.pi], device=self.device), sigma**2))), \
               torch.exp(-torch.div((x - mu)**2, 2 * sigma**2))).mul(_pi)

    def GMM_region_responsibility(self, weights):
        """" Region responsibility of GMM. """
        pi_normalized = self.gaussian_mixing_regularization().cuda()
        responsibility = torch.zeros([self.num_components, weights.size(0)], device=self.device)
        responsibility[0] = self.Normal_pdf(weights.cuda(), pi_normalized[0], 0.0, self.sigma_zero.cuda())
        for k in range(self.num_components-1):
            responsibility[k+1] = self.Normal_pdf(weights.cuda(), pi_normalized[k+1], self.mu[k].cuda(), self.sigma[k].cuda())
        responsibility = torch.div(responsibility, responsibility.sum(dim=0) + cfg.EPS)
        return F.softmax(responsibility / self.temperature, dim=0)

    def forward(self, weights, train=True):
        if train:
            # soft mask generalized pruning during training
            self.region_belonging = self.GMM_region_responsibility(weights.flatten())
            Sweight = torch.mul(self.region_belonging[0], 0.) \
                    + torch.mul(self.region_belonging[1:], self.mu.unsqueeze(1)).sum(dim=0)
            return Sweight.view(weights.size())
        else:
            self.region_belonging = self.GMM_region_responsibility(weights.flatten())
            max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
            mask_w = torch.zeros_like(self.region_belonging).scatter_(dim=0, index=max_index, value=1.)
            Pweight = torch.mul(mask_w[1:], self.mu.unsqueeze(1)).sum(dim=0)
            return Pweight.view(weights.size())

def gmm_approximation(num_components, init_weights, temperature=0.5, init_method='k-means'):
    return GaussianMixtureModel(num_components, init_weights.flatten(), temperature, init_method)
