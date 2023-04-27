import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument('--dataset', type = str, default = 'cifar10',  help = 'name of the dataset: mnist, cifar10' )
    parser.add_argument( '--model', type = str, default = 'cnn',
                         help='name of model. mnist: logistic, lenet; cifar10: cnn_tutorial, cnn_complex.')
    parser.add_argument( '--train_ratio', default=1.0, type=float, help='dataset train ratio' )
    parser.add_argument( '--input_channels', type = int, default = 3, help = 'input channels. mnist:1, cifar10 :3' )
    parser.add_argument( '--output_channels', type = int, default = 10, help = 'output channels' )

    #fednn training hyper parameter
    parser.add_argument( '--bs', type = int, default = 10, help = 'batch size for local update on private data' )
    parser.add_argument( '--local_step', type = int, default= 1, help= 'local update on private data' )
    parser.add_argument( '--num_communication', type = int, default=1, help = 'number of communication rounds with the cloud server' )
    parser.add_argument( '--optimizer', type=str, default='SGD', help='optimizer for the client and server.' )
    parser.add_argument( '--lr_scheduler', type=str, default= None, help='lr_scheduler.' )
    parser.add_argument( '--lr', type = float, default = 0.001, help = 'learning rate of the SGD of local update' )
    parser.add_argument( '--lr_decay', type = float, default= 0.99, help = 'lr decay rate' )
    parser.add_argument( '--lr_decay_epoch', type = int, default=1, help= 'lr decay epoch' )
    parser.add_argument( '--momentum', type = float, default = 0, help = 'SGD momentum' )
    parser.add_argument( '--weight_decay', type = float, default = 0, help= 'The weight decay rate' )
    parser.add_argument( '--verbose', type = bool, default = False, help = 'verbose for print progress bar' )

    #setting for data distribution
    parser.add_argument( '--iid', type = int, default = 1, help = 'distribution of the data, 1,0, -2(one-class)' )
    parser.add_argument( '--alpha', type=float, default=1.0, help='param of the dirichlet distribution, only valid when --iid==-1' )
    parser.add_argument( '--double_stochastic', type=int, default=1, help='mking the dirichlet double stochastic' )
    parser.add_argument( '--dataset_root', type=str,  default='data', help='dataset root folder' )
    parser.add_argument( '--show_dis', type=int, default=0, help='whether to show distribution' )
    parser.add_argument( '--classes_per_client', type=int, default=2, help='under artificial non-iid distribution, the classes per client' )
    # Params for FL sys
    parser.add_argument( '--client_frac', type=float, default=1, help='fraction of participated clients' )
    parser.add_argument( '--num_clients', type = int, default = 2, help = 'number of all available clients' )
    parser.add_argument( '--seed', type = int, default = 1, help = 'random seed (defaul: 1)' )

    parser.add_argument( '--gpu', type = int, default=0, help = 'GPU to be selected, 0, 1, 2, 3' )

    # Params for dp-mechanism
    parser.add_argument( '--dp_mechanism', type = str, default= 'no_dp', help = 'no_dp, laplace, gaussian,rr' )
    parser.add_argument( '--dp_epsilon', type= float, default = 1.0, )
    parser.add_argument( '--dp_delta', type = float, default=0.0 )
    parser.add_argument( '--dp_clip', type = float, default=100.0 )
    parser.add_argument('--dp_pattern', type=str, default='UNIFORM')
    parser.add_argument('--dp_decay', type=float, default= 1.0)
    parser.add_argument('--dp_flip', type=float, default=0.0)

    # Binary
    parser.add_argument('--comm_mode', type=str, default='full')
    parser.add_argument('--sigma', type=float, default=0.2)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
