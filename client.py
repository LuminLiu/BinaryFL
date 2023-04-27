import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy

def local_train(model, optimizer, local_step, trainloader, criterion, args):
    model.train()
    for i, data in enumerate(trainloader): #assuming local_step <= num_sample/bs, local update no more than 1 epoch
        # get the inputs; data is a list of [inputs, labels]
        if i > local_step:
            break
        inputs, labels = data
        inputs = Variable(inputs).to(args.device)
        labels = Variable(labels).to(args.device)
        model.zero_grad()
        outputs = model(inputs) #forward
        loss = criterion(outputs, labels)
        loss.backward() #compute gradients
        if 'binary' in args.model:
            for p in model.parameters():
                if hasattr(p, 'org'):
                    p.data.copy_(p.org) #org is the auxiliary parameters
        optimizer.step()
        if 'binary' in args.model:
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))
    return None

def pre_com(comm_mode, model):
    for cv in model.parameters():
        if hasattr(cv, 'org'):
            if comm_mode == 'bin':
                cv.data.sign_()
                # maxtheta=torch.max(cv.data.abs())
                # cv.data.mul_(1.49/maxtheta).round_()
                # print(cv.data)
                # exit()
            elif comm_mode == 'full':
                # print('transmitting full-precision weights')
                cv.data.copy_(cv.org)
            else:
                print("Undefined mode!")

def adding_noise(model, args):
    if args.dp_mechanism != 'no_dp':
        # print('adding_noise')
        for cv in model.parameters():
            if hasattr(cv, 'org'):
                if np.random.uniform(0, 1, 1) < args.dp_flip:
                    cv.data = cv.data.mul_(-1)


def local_update(model, weights, args):
    model.load_state_dict(copy.deepcopy(weights))
    for cv in model.parameters():
        if hasattr(cv, 'org'):
            cv.org.mul_(1.0 - args.sigma).add_(cv.data.mul_(args.sigma))  # \sigma is the \beta in the paper

