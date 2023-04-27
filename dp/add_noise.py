import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import random
from sklearn import metrics

def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size
#     return 2 * lr * clip

def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)

# todo
def Gaussian_moment(epsilon, delta, sensitivity, size):
    return

def clip_gradients( net, args):
    if args.dp_mechanism == 'Laplace':
        # Laplace use 1 norm
        perSampleClip(net, args.dp_clip, norm=1)
    elif args.dp_mechanism == 'Gaussian':
        # Gaussian use 2 norm
        perSampleClip(net, args.dp_clip, norm=2)

def perSampleClip(net, clipping, norm):
    grad_samples = [x.grad_sample for x in net.parameters()]
    per_param_norms = [
        g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
    ]
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
    per_sample_clip_factor = (
        torch.div(clipping, (per_sample_norms + 1e-6))
    ).clamp(max=1.0)
    for factor, grad in zip(per_sample_clip_factor, grad_samples):
        grad.detach().mul_(factor.to(grad.device))
    # average per sample gradient after clipping and set back gradient
    for param in net.parameters():
        param.grad = param.grad_sample.detach().mean(dim=0)

def add_noise(net, args, idxs):
    sensitivity = cal_sensitivity(args.lr, args.dp_clip, len(idxs))
    if args.dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Laplace(epsilon=args.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(args.device)
                v += noise
    elif args.dp_mechanism == 'Gaussian':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=args.dp_epsilon, delta=args.dp_delta, sensitivity=sensitivity,
                                        size=v.shape)
                noise = torch.from_numpy(noise).to(args.device)
                v += noise



