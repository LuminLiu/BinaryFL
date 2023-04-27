use different learning rate

mlp_binary for mnist: decays by 0.1 every 40 epochs, bs=64, 100 epochs 98.4%acc 

resnet_binary for cifar10: \
self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }

