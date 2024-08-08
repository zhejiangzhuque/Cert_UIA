import os
import sys
sys.path.append(os.path.dirname(__file__))

from training import train_gcn, pretrain_F, train_uinj


if __name__ == '__main__':
    train_gcn.run()
    train_gcn.test()
    pretrain_F.run()
    train_uinj.run()
    train_uinj.test()
    train_uinj.exp()