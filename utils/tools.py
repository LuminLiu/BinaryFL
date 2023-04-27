import resource
def monitor_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(usage)
    exit()

# def adapt_dp(args, pattern, comm, mean, l_bound):
#     if pattern == 'ASCENDING':
#         step = (mean-l_bound)*2/args.num_communication
#         args.dp_epsilon = l_bound + comm*step
#     elif pattern == 'DESCENDING':
#         u_bound = 2 * mean - l_bound
#         step = (mean - l_bound) * 2 / args.num_communication
#         args.dp_epsilon = u_bound - comm*step
#     return None

def adapt_dp(args):
    args.dp_epsilon = args.dp_epsilon * args.dp_decay
    return None

def modify_optimizer(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# def adjust_optimizer(optimizer, epoch): #mnist
#     if epoch == 0:
#         modify_optimizer(optimizer, 5e-3)
#     elif epoch == 30:
#         modify_optimizer(optimizer, 2e-3)
#     elif epoch == 60:
#         modify_optimizer(optimizer, 1e-3)

# def adjust_optimizer(optimizer, epoch): #cifar,vgg
#     if epoch == 0:
#         modify_optimizer(optimizer, 5e-3)
#     elif epoch == 40:
#         modify_optimizer(optimizer, 1e-3)
#     elif epoch == 80:
#         modify_optimizer(optimizer, 5e-4)
#     elif epoch == 100:
#         modify_optimizer(optimizer, 1e-4)
#     elif epoch == 120:
#         modify_optimizer(optimizer, 5e-5)
#     elif epoch == 140:
#         modify_optimizer(optimizer, 1e-5)

def adjust_optimizer(optimizer, epoch):  # cifar,resnet18
    if epoch == 0:
        modify_optimizer(optimizer, 5e-3)
    elif epoch == 101:
        modify_optimizer(optimizer, 1e-3)
    elif epoch == 142:
        modify_optimizer(optimizer, 5e-4)
    elif epoch == 184:
        modify_optimizer(optimizer, 1e-4)
    elif epoch == 220:
        modify_optimizer(optimizer, 1e-5)



