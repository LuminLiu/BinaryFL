# Specify Data Path
import copy
from utils.tools import monitor_usage
DATAPATH_ROOT = "/home/lliubb/Data"

from tensorboardX import SummaryWriter
from options import args_parser
from datasets.get_data import get_dataset
from fednn.initialize import initialize_nn
import torch
import random
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
from dp.add_noise import clip_gradients, perSampleClip, add_noise
from opacus.grad_sample import GradSampleModule
from utils.tools import adjust_optimizer

def test(net, test_loader, device, writer, epoch):
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
    writer.add_scalar(f'test_acc',
                      test_acc,
                      epoch + 1)
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

    args.dataset = 'mnist'  # 'mnist'
    # args.model = 'mnistnet_binary'  # 'lenet, logistic, resnet18, resnet8'
    args.model = 'mlp_binary'
    # args.dataset = 'cifar10'
    # args.model = 'vgg_cifar10_binary'
    # args.dataset = 'cifar10'
    # args.model = 'resnet_binary'
    args.num_clients = 1
    args.num_communication = 50
    args.bs = 64
    args.lr = 0.01
    args.momentum= 0.5
    args.weight_decay = 1e-4
    args.iid = 1
    # dp parameters
    args.dp_mechanism = 'no_dp'
    args.dp_epsilon = 10 / args.num_communication
    args.dp_delta = 1e-5
    args.dp_clip = 10

    args.verbose = True

    if args.dp_mechanism == 'no_dp':
        FILEOUT = f'1Client{args.dp_mechanism}-{args.num_communication}epochs-{args.dataset}-{args.model}'
        print(FILEOUT)
    elif args.dp_mechanism == 'Laplace':
        FILEOUT = f'1Client{args.dp_mechanism}-{args.dp_epsilon}-{args.num_communication}epochs-{args.dataset}-{args.model}'
        print(FILEOUT)
    elif args.dp_mechanism == 'Gaussian':
        FILEOUT = f'1Client{args.dp_mechanism}-{args.dp_epsilon}-{args.dp_delta}-{args.num_communication}epochs-{args.dataset}-{args.model}'
        print(FILEOUT)
    else:
        raise ValueError(f'{args.dp_mechanism} is NOT implemented yet.')
    if args.verbose:
        writer = SummaryWriter(comment=FILEOUT)

    model = initialize_nn(args, device)
    train_loaders, test_loaders, _ = get_dataset(dataset_root=DATAPATH_ROOT,
                                               dataset=args.dataset,
                                               args=args)
    net = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params= net.parameters(),
    #                        lr= args.lr,
    #                        betas=(0.9, 0.999),
    #                        weight_decay=args.weight_decay
    # )
    optimizer =optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.SGD(params=net.parameters(),
    #                       lr=args.lr,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay
    #                       )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    net.train()
    print(net)
    # monitor_usage()

    running_loss = 0.0
    for epoch in tqdm(range(100)): # loop over the dataset multiple times
        # adjust_optimizer(optimizer, epoch)
        if epoch%40==39:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1
            current_lr = optimizer.param_groups[0]['lr']
            print(f'adjust lr as {current_lr} at epoch{epoch}')
        for i, data in enumerate(train_loaders[0]):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if 'binary' in args.model:
                for p in net.parameters():
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)
            optimizer.step()
            if 'binary' in args.model:
                for p in list(net.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1,1))
            if args.dp_mechanism != 'no_dp':
                add_noise(net=net, args=args, idxs=train_loaders[0].dataset.idxs)
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        if args.verbose:
            test(net, test_loaders[0], device, writer, epoch)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

