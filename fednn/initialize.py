# from fednn.cifar_cnn import cifar_cnn_3conv
from fednn.resnet import ResNet18
# from fednn.fdresnet import ResNet_cifar
# from fednn.lenet import mnist_lenet
# from fednn.logi import LogisticRegression
from fednn.mnist_binary import mnistnet_binary
from fednn.mlp_binary import mlp_binary
# from fednn.vgg_cifar10_binary import VGG_Cifar10
from fednn.resnet_binary import resnet_binary


# following import is used for testing the function of this part, they can be deleted if you delete the main() funciton
# from options import args_parser
# from datasets.get_data import get_dataset
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from os.path import dirname, abspath, join
# from torch.autograd import Variable
# from tqdm import tqdm
# DATAPATH_ROOT = "/home/lliubb/Data"
# sys.path.insert(0, "../")
# sys.path.insert(0, "../../")

def initialize_nn(args, device):
    if args.dataset == 'cifar10':
        if args.model == 'cnn_complex':
            localmodel = cifar_cnn_3conv(input_channels=3, output_channels=10)
        elif args.model == 'resnet18':
            localmodel = ResNet18()
        elif args.model == 'resnet8':
            localmodel = ResNet_cifar(dataset="cifar10",
                                      resnet_size=8,
                                      group_norm_num_groups=None,
                                      freeze_bn=False,
                                      freeze_bn_affine=False)
        elif args.model == 'vgg_cifar10_binary':
            localmodel = VGG_Cifar10()
        elif args.model == 'resnet_binary':
            localmodel = resnet_binary()
        else:
            raise ValueError('NN not implemented for cifar-10, please check your input.')
    elif args.dataset == 'mnist' or args.dataset == 'fashionmnist':
        if args.model == 'lenet':
            localmodel = mnist_lenet(input_channels=1, output_channels=10)
        elif args.model == 'logistic':
            localmodel = LogisticRegression(input_dim=1, output_dim=10)
        elif args.model == 'mnistnet_binary':
            localmodel = mnistnet_binary()
        elif args.model == 'mlp_binary':
            localmodel = mlp_binary()
        else:
            raise ValueError('NN not implemented for mnist, please check your input.')
    else:
        raise ValueError('Current codes have not inluded this dataset yet, please check your input.')
    if args.cuda:
        localmodel = localmodel.cuda(device)
    return localmodel



# def main():
#     args = args_parser()
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#         cuda_to_use = torch.device(f'cuda:{args.gpu}')
#     device = cuda_to_use if torch.cuda.is_available() else "cpu"
#
#     args.dataset = 'cifar10'  #'mnist'
#     args.model = 'cnn_complex' #'lenet, logistic, resnet18, resnet8'
#     model = initialize_nn(args, device)
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#          ]
#     )
#     transform_train = transform
#     transform_test = transform
#     trainset = torchvision.datasets.CIFAR10(root=DATAPATH_ROOT,train=True, download=True, transform= transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset,batch_size= args.bs_u, shuffle=True,num_workers = 0)
#     testset = torchvision.datasets.CIFAR10(root=DATAPATH_ROOT, train=False,
#                                            download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs_u,
#                                              shuffle=False, num_workers=0)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(params= model.parameters(),
#                           lr = args.eta_u,
#                           momentum=args.momentum,
#                           weight_decay=args.weight_decay
#     )
#     for epoch in tqdm(range(2)):  # loop over the dataset multiple times
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs = Variable(inputs).to(device)
#             labels = Variable(labels).to(device)
#             outputs = model(inputs)
#             optimizer.zero_grad()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#
#     print('Finished Training')
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         model.train(False)
#         for data in testloader:
#             images, labels = data
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#             100 * correct / total))
#
#
# if __name__ == '__main__':
#     main()