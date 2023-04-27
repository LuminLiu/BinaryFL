import copy

DATAPATH_ROOT = "/home/lliubb/Data"
from tensorboardX import SummaryWriter
from options import args_parser
from datasets.get_data import get_dataset
from fednn.initialize import initialize_nn

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np

from client import local_train, pre_com, adding_noise, local_update
from utils.avg import average_weights

def test(net, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        net.train(False)
        for data in test_loader:
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    net.train()
    return test_acc


def main():
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # for the reproducibility, but the performance might reduce
    torch.backends.cudnn.benchmark = False
    args.dataset_root = DATAPATH_ROOT
    args.device = device

    # args.dataset = 'mnist'  # 'mnist'
    # args.model = 'mnistnet_binary'  # 'lenet, logistic, resnet18, resnet8'
    # args.model = 'mlp_binary'
    # args.dataset = 'cifar10'
    # args.model = 'vgg_cifar10_binary'
    # args.dataset = 'cifar10'
    # args.model = 'resnet_binary'
    args.num_clients = 100
    args.num_communication = 100
    args.bs = 64
    args.local_step = 10
    args.lr = 0.01
    args.momentum = 0.5
    args.weight_decay = 1e-4
    # args.iid = 1

    # dp parameters
    # args.dp_mechanism = 'no_dp'
    # args.dp_mechanism = 'rr'
    # args.dp_flip = 0.2
    # for gaussian
    # args.dp_epsilon = 10 / args.num_communication
    # args.dp_delta = 1e-5
    # args.dp_clip = 10

    # args.comm_mode = 'full'
    # args.sigma = 1.0
    args.comm_mode = 'bin'
    args.sigma = 0.2
    args.verbose = True

    if args.dp_mechanism == 'no_dp':
        FILEOUT = f'unbiased{args.num_clients}Client{args.dp_mechanism}-{args.num_communication}epochs-{args.dataset}-{args.model}'
        print(FILEOUT)
    elif args.dp_mechanism == 'rr':
        FILEOUT = f'unbiased{args.num_clients}Client{args.dp_mechanism}flip{args.dp_flip}-{args.num_communication}epochs-{args.dataset}-{args.model}'
    if args.verbose:
        writer = SummaryWriter(comment=FILEOUT)
    server_net = initialize_nn(args, device)
    train_loaders, _, v_test_loader = get_dataset(DATAPATH_ROOT,args.dataset, args)
    sample_num = [len(train_loader.dataset) for train_loader in train_loaders]
    # print(f'sample number of first client is {sample_num[0]}')
    # exit()
    #Initialize clients, not including sampling now
    criterion = nn.CrossEntropyLoss()
    client_nets = []
    optimizers=[]
    best_acc = 0.0
    for i in range(args.num_clients):
        local_model = copy.deepcopy(server_net)
        optimizer = optim.Adam(local_model.parameters(), args.lr)
        client_nets.append(local_model)
        optimizers.append(optimizer)
    for comm in tqdm(range(args.num_communication)):
        #sequentially excute local training
        received_weights = []
        for cid in range(args.num_clients):
            # adjust lr if necessary
            if comm % 40 == 39:
                optimizers[cid].param_groups[0]['lr'] = optimizers[cid].param_groups[0]['lr'] * 0.1
                current_lr = optimizers[cid].param_groups[0]['lr']
                print(f'adjust lr as {current_lr} for client{cid} at epoch{comm}')
            local_train(model=client_nets[cid],optimizer=optimizers[cid],
                        local_step=args.local_step,
                        trainloader=train_loaders[cid],
                        criterion=criterion,args=args)
            pre_com(comm_mode=args.comm_mode, model=client_nets[cid])
            adding_noise(client_nets[cid], args)
            received_weights.append(copy.deepcopy(client_nets[cid].state_dict()))
        aggregated_weights = average_weights(received_weights, sample_num)
        # for the unbiased estimation
        if args.dp_mechanism == 'rr':
            double_gamma = 1- 2*args.dp_flip
            for k in aggregated_weights.keys():
                aggregated_weights[k] = torch.div(aggregated_weights[k], double_gamma)

        for cid in range(args.num_clients):
            local_update(model=client_nets[cid],
                         weights= aggregated_weights,
                         args=args)
        if args.verbose:
            # server_net.load_state_dict(aggregated_weights)
            # if 'binary' in args.model:
            #     for p in server_net.parameters():
            #         if hasattr(p, 'org'):
            #             p.org.copy_(p.data)
            # test_acc = test(server_net, v_test_loader, device)
            # print(f'Testing acc of servernet is{test_acc}')
            # test_acc = test(client_nets[0], v_test_loader, device)
            # print(f'Testing acc of clientnet is {test_acc}')
            accu_test_acc = 0.0
            test_subset = np.random.choice(args.num_clients, 10)
            for cid in test_subset:
                accu_test_acc += test(client_nets[cid], v_test_loader, device)
            test_acc = accu_test_acc / args.num_clients
            writer.add_scalar(f'test_acc_avg',
                              test_acc,
                              comm + 1)
            if test_acc > best_acc:
                best_acc = test_acc
                best_comm = comm
                print(f'best_acc {best_acc} is obtained at {best_comm}')
    print(f'best_acc {best_acc} is obtained at {best_comm}')
if __name__ == '__main__':
    main()




